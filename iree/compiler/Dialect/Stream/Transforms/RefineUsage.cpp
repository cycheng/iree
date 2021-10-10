// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Element.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-stream-refine-usage"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {
namespace {

//===----------------------------------------------------------------------===//
// Resource usage analysis
//===----------------------------------------------------------------------===//

enum class ResourceUsageBitfield : uint32_t {
  Indirect = 1u << 0,
  External = 1u << 1,
  Mutated = 1u << 2,  // beyond definition
  Constant = 1u << 3,
  TransferRead = 1u << 4,
  TransferWrite = 1u << 5,
  StagingRead = 1u << 6,
  StagingWrite = 1u << 7,
  DispatchRead = 1u << 8,
  DispatchWrite = 1u << 9,
  GlobalRead = 1u << 10,
  GlobalWrite = 1u << 11,

  Unknown = Indirect | External | Mutated | Constant | TransferRead |
            TransferWrite | StagingRead | StagingWrite | DispatchRead |
            DispatchWrite | GlobalRead | GlobalWrite,
};
inline ResourceUsageBitfield operator|(ResourceUsageBitfield lhs,
                                       ResourceUsageBitfield rhs) {
  return static_cast<ResourceUsageBitfield>(static_cast<uint32_t>(lhs) |
                                            static_cast<uint32_t>(rhs));
}
inline ResourceUsageBitfield operator&(ResourceUsageBitfield lhs,
                                       ResourceUsageBitfield rhs) {
  return static_cast<ResourceUsageBitfield>(static_cast<uint32_t>(lhs) &
                                            static_cast<uint32_t>(rhs));
}
inline bool bitEnumContains(ResourceUsageBitfield bits,
                            ResourceUsageBitfield bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}

static Lifetime convertUsageToLifetime(ResourceUsageBitfield usage) {
  if (bitEnumContains(usage, ResourceUsageBitfield::Indirect) ||
      bitEnumContains(usage, ResourceUsageBitfield::External)) {
    return Lifetime::External;
  } else if (bitEnumContains(usage, ResourceUsageBitfield::StagingRead) ||
             bitEnumContains(usage, ResourceUsageBitfield::StagingWrite)) {
    return Lifetime::Staging;
  } else if (bitEnumContains(usage, ResourceUsageBitfield::Constant)) {
    return Lifetime::Constant;
  } else if (bitEnumContains(usage, ResourceUsageBitfield::GlobalRead) ||
             bitEnumContains(usage, ResourceUsageBitfield::GlobalWrite)) {
    return bitEnumContains(usage, ResourceUsageBitfield::Mutated) ||
                   bitEnumContains(usage, ResourceUsageBitfield::GlobalWrite) ||
                   bitEnumContains(usage,
                                   ResourceUsageBitfield::DispatchWrite) ||
                   bitEnumContains(usage,
                                   ResourceUsageBitfield::StagingWrite) ||
                   bitEnumContains(usage, ResourceUsageBitfield::TransferWrite)
               ? Lifetime::Variable
               : Lifetime::Constant;
  } else {
    return Lifetime::Transient;
  }
}

// Starts by assuming that the resource is never used and then removes assumed
// bits based on the usage in the program.
//
// Best state: never used at all.
// Worst state: used for all kinds of things.
template <typename ElementT>
class AbstractResourceUsage
    : public DFX::StateWrapper<DFX::BitIntegerState<uint16_t, 4095, 0>,
                               ElementT> {
 public:
  using BaseType =
      DFX::StateWrapper<DFX::BitIntegerState<uint16_t, 4095, 0>, ElementT>;

  enum {
    NOT_INDIRECT = 1u << 0,
    NOT_EXTERNAL = 1u << 1,
    NOT_MUTATED = 1u << 2,  // beyond definition
    NOT_CONSTANT = 1u << 3,
    NOT_TRANSFER_READ = 1u << 4,
    NOT_TRANSFER_WRITE = 1u << 5,
    NOT_STAGING_READ = 1u << 6,
    NOT_STAGING_WRITE = 1u << 7,
    NOT_DISPATCH_READ = 1u << 8,
    NOT_DISPATCH_WRITE = 1u << 9,
    NOT_GLOBAL_READ = 1u << 10,
    NOT_GLOBAL_WRITE = 1u << 11,

    BEST_STATE = NOT_INDIRECT | NOT_EXTERNAL | NOT_MUTATED | NOT_CONSTANT |
                 NOT_TRANSFER_READ | NOT_TRANSFER_WRITE | NOT_STAGING_READ |
                 NOT_STAGING_WRITE | NOT_DISPATCH_READ | NOT_DISPATCH_WRITE |
                 NOT_GLOBAL_READ | NOT_GLOBAL_WRITE,
  };
  static_assert(BEST_STATE == BaseType::getBestState(),
                "unexpected BEST_STATE value");

  ResourceUsageBitfield convertBitsToResourceUsage(uint16_t bits) const {
    return static_cast<ResourceUsageBitfield>(~bits & BEST_STATE);
  }

  ResourceUsageBitfield getKnownUsage() const {
    return convertBitsToResourceUsage(this->getKnown());
  }

  ResourceUsageBitfield getAssumedUsage() const {
    return convertBitsToResourceUsage(this->getAssumed());
  }

  const std::string getAsStr() const override {
    std::string str;
    auto append = [&](const char *part) {
      if (!str.empty()) str += '|';
      str += part;
    };
    if (!this->isAssumed(NOT_INDIRECT)) append("indirect");
    append(this->isAssumed(NOT_EXTERNAL) ? "internal" : "external");
    append(this->isAssumed(NOT_MUTATED) ? "immutable" : "mutable");
    if (!this->isAssumed(NOT_CONSTANT)) append("constant");
    if (!this->isAssumed(NOT_TRANSFER_READ)) append("transfer_read");
    if (!this->isAssumed(NOT_TRANSFER_WRITE)) append("transfer_write");
    if (!this->isAssumed(NOT_STAGING_READ)) append("staging_read");
    if (!this->isAssumed(NOT_STAGING_WRITE)) append("staging_write");
    if (!this->isAssumed(NOT_DISPATCH_READ)) append("dispatch_read");
    if (!this->isAssumed(NOT_DISPATCH_WRITE)) append("dispatch_write");
    if (!this->isAssumed(NOT_GLOBAL_READ)) append("global_read");
    if (!this->isAssumed(NOT_GLOBAL_WRITE)) append("global_write");
    return str.empty() ? "*" : str;
  }

 protected:
  explicit AbstractResourceUsage(const Position &pos) : BaseType(pos) {}

  void initializeFromType(IREE::Stream::ResourceType type) {
    BaseType::intersectAssumedBits(BEST_STATE);
    switch (type.getLifetime()) {
      case Lifetime::Unknown:
        break;
      case Lifetime::External:
        BaseType::addKnownBits(NOT_CONSTANT | NOT_STAGING_READ |
                               NOT_STAGING_WRITE);
        break;
      case Lifetime::Staging:
        BaseType::addKnownBits(NOT_EXTERNAL | NOT_CONSTANT | NOT_DISPATCH_READ |
                               NOT_DISPATCH_WRITE | NOT_GLOBAL_READ |
                               NOT_GLOBAL_WRITE);
        break;
      case Lifetime::Transient:
        BaseType::addKnownBits(NOT_EXTERNAL | NOT_CONSTANT | NOT_STAGING_READ |
                               NOT_STAGING_WRITE);
        break;
      case Lifetime::Variable:
        BaseType::addKnownBits(NOT_EXTERNAL | NOT_CONSTANT | NOT_STAGING_READ |
                               NOT_STAGING_WRITE);
        break;
      case Lifetime::Constant:
        BaseType::addKnownBits(NOT_MUTATED | NOT_EXTERNAL | NOT_STAGING_READ |
                               NOT_STAGING_WRITE);
        break;
    }
  }
};

// Starts with the best assumed state of the value never being used for anything
// and then works towards a worst state of it being used for everything.
class ValueResourceUsage : public AbstractResourceUsage<DFX::ValueElement> {
 public:
  using BaseType = AbstractResourceUsage<DFX::ValueElement>;

  static ValueResourceUsage &createForPosition(const Position &pos,
                                               DFX::Solver &solver) {
    return *(new (solver.getAllocator()) ValueResourceUsage(pos));
  }

  const std::string getName() const override { return "ValueResourceUsage"; }
  const char *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }

  static const char ID;

 private:
  explicit ValueResourceUsage(const Position &pos) : BaseType(pos) {}

  void initializeValue(Value value, DFX::Solver &solver) override {
    auto resourceType = value.getType().cast<IREE::Stream::ResourceType>();
    initializeFromType(resourceType);
  }

  void updateFromDefiningOp(Value value, OpResult result, DFX::Solver &solver) {
    TypeSwitch<Operation *, void>(result.getOwner())
        .Case<IREE::Util::DoNotOptimizeOp>([&](auto op) {
          auto sourceUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getOperand(0)),
              DFX::Resolution::REQUIRED);
          getState() ^= sourceUsage.getState();
        })
        .Case<IREE::Util::GlobalLoadOp>([&](auto op) {
          removeAssumedBits(NOT_GLOBAL_READ);
          auto *globalInfo =
              solver.getExplorer().queryGlobalInfoFrom(op.global(), op);
          auto globalType =
              globalInfo->op.type().template cast<IREE::Stream::ResourceType>();
          switch (globalType.getLifetime()) {
            case IREE::Stream::Lifetime::Constant:
              removeAssumedBits(NOT_CONSTANT);
              break;
            case IREE::Stream::Lifetime::Variable:
            default:
              break;
          }
        })
        .Case<IREE::Util::GlobalLoadIndirectOp>(
            [&](auto op) { removeAssumedBits(NOT_INDIRECT | NOT_GLOBAL_READ); })
        .Case<IREE::Stream::ResourceStoreOp>([&](auto op) {
          removeAssumedBits(NOT_STAGING_WRITE);
          auto targetUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.target()),
              DFX::Resolution::REQUIRED);
          getState() ^= targetUsage.getState();
        })
        .Case<IREE::Stream::TensorImportOp>(
            [&](auto op) { removeAssumedBits(NOT_MUTATED | NOT_EXTERNAL); })
        .Case<IREE::Stream::AsyncConstantOp>([&](auto op) {
          removeAssumedBits(NOT_CONSTANT | NOT_TRANSFER_WRITE);
        })
        .Case<IREE::Stream::AsyncSplatOp>(
            [&](auto op) { removeAssumedBits(NOT_TRANSFER_WRITE); })
        .Case<IREE::Stream::AsyncCloneOp>([&](auto op) {
          removeAssumedBits(NOT_TRANSFER_WRITE);
          auto sourceUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.source()),
              DFX::Resolution::OPTIONAL);
          getState() ^= sourceUsage.getState();
        })
        .Case<IREE::Stream::AsyncSliceOp>([&](auto op) {
          removeAssumedBits(NOT_TRANSFER_WRITE);
          auto sourceUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.source()),
              DFX::Resolution::OPTIONAL);
          getState() ^= sourceUsage.getState();
        })
        .Case<IREE::Stream::AsyncFillOp>([&](auto op) {
          removeAssumedBits(NOT_TRANSFER_WRITE);
          auto targetUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.target()),
              DFX::Resolution::REQUIRED);
          getState() ^= targetUsage.getState();
        })
        .Case<IREE::Stream::AsyncUpdateOp>([&](auto op) {
          removeAssumedBits(NOT_TRANSFER_WRITE);
          auto targetUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.target()),
              DFX::Resolution::REQUIRED);
          getState() ^= targetUsage.getState();
        })
        .Case<IREE::Stream::AsyncCopyOp>([&](auto op) {
          removeAssumedBits(NOT_TRANSFER_WRITE);
          auto targetUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.target()),
              DFX::Resolution::REQUIRED);
          getState() ^= targetUsage.getState();
        })
        .Case<IREE::Stream::AsyncTransferOp>([&](auto op) {
          removeAssumedBits(NOT_TRANSFER_WRITE);
          auto sourceUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.source()),
              DFX::Resolution::OPTIONAL);
          bool isSourceStaging = !(sourceUsage.isAssumed(NOT_STAGING_READ) &&
                                   sourceUsage.isAssumed(NOT_STAGING_WRITE));
          bool isTargetStaging =
              !(isAssumed(NOT_STAGING_READ) && isAssumed(NOT_STAGING_WRITE));
          if (isSourceStaging != isTargetStaging) {
            // Can't transition staging across transfers.
            LLVM_DEBUG({
              llvm::dbgs() << "[ValueResourceUsage] skipping transfer source: ";
              op.print(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << "\n";
            });
            return;
          }
          getState() ^= sourceUsage.getState();
        })
        .Case<IREE::Stream::AsyncDispatchOp>([&](auto op) {
          removeAssumedBits(NOT_DISPATCH_WRITE);
          auto tiedOperand = op.getTiedResultOperand(result);
          if (tiedOperand) {
            auto tiedUsage = solver.getElementFor<ValueResourceUsage>(
                *this, Position::forValue(tiedOperand),
                DFX::Resolution::REQUIRED);
            getState() ^= tiedUsage.getState();
          }
        })
        .Default([&](Operation *op) {});
  }

  void updateFromUse(Value value, OpOperand &operand, DFX::Solver &solver) {
    auto *userOp = operand.getOwner();
    unsigned operandIdx = operand.getOperandNumber();
    TypeSwitch<Operation *, void>(userOp)
        .Case<IREE::Util::DoNotOptimizeOp>([&](auto op) {
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getResult(0)),
              DFX::Resolution::REQUIRED);
          getState() ^= resultUsage.getState();
        })
        .Case<IREE::Util::GlobalStoreOp>([&](auto op) {
          removeAssumedBits(NOT_GLOBAL_WRITE);
          auto *globalInfo =
              solver.getExplorer().queryGlobalInfoFrom(op.global(), op);
          auto globalType =
              globalInfo->op.type().template cast<IREE::Stream::ResourceType>();
          switch (globalType.getLifetime()) {
            case IREE::Stream::Lifetime::Constant:
              removeAssumedBits(NOT_CONSTANT);
              break;
            case IREE::Stream::Lifetime::Variable:
            default:
              break;
          }
        })
        .Case<IREE::Util::GlobalStoreIndirectOp>([&](auto op) {
          removeAssumedBits(NOT_INDIRECT | NOT_GLOBAL_WRITE);
        })
        .Case<IREE::Stream::ResourceLoadOp>(
            [&](auto op) { removeAssumedBits(NOT_STAGING_READ); })
        .Case<IREE::Stream::ResourceStoreOp>([&](auto op) {
          removeAssumedBits(NOT_MUTATED | NOT_STAGING_WRITE);
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.result()),
              DFX::Resolution::REQUIRED);
          getState() ^= resultUsage.getState();
        })
        .Case<IREE::Stream::TensorExportOp>(
            [&](auto op) { removeAssumedBits(NOT_MUTATED | NOT_EXTERNAL); })
        .Case<IREE::Stream::AsyncCloneOp>([&](auto op) {
          removeAssumedBits(NOT_TRANSFER_READ);
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.result()),
              DFX::Resolution::OPTIONAL);
          getState() ^= resultUsage.getState();
        })
        .Case<IREE::Stream::AsyncSliceOp>([&](auto op) {
          removeAssumedBits(NOT_TRANSFER_READ);
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.result()),
              DFX::Resolution::OPTIONAL);
          getState() ^= resultUsage.getState();
        })
        .Case<IREE::Stream::AsyncFillOp>([&](auto op) {
          removeAssumedBits(NOT_MUTATED | NOT_TRANSFER_WRITE);
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.result()),
              DFX::Resolution::REQUIRED);
          getState() ^= resultUsage.getState();
        })
        .Case<IREE::Stream::AsyncUpdateOp>([&](auto op) {
          if (value == op.update()) {
            removeAssumedBits(NOT_TRANSFER_READ);
          } else {
            removeAssumedBits(NOT_MUTATED | NOT_TRANSFER_WRITE);
            auto resultUsage = solver.getElementFor<ValueResourceUsage>(
                *this, Position::forValue(op.result()),
                DFX::Resolution::REQUIRED);
            getState() ^= resultUsage.getState();
          }
        })
        .Case<IREE::Stream::AsyncCopyOp>([&](auto op) {
          if (value == op.source()) {
            removeAssumedBits(NOT_TRANSFER_READ);
          } else {
            removeAssumedBits(NOT_MUTATED | NOT_TRANSFER_WRITE);
            auto resultUsage = solver.getElementFor<ValueResourceUsage>(
                *this, Position::forValue(op.result()),
                DFX::Resolution::REQUIRED);
            getState() ^= resultUsage.getState();
          }
        })
        .Case<IREE::Stream::AsyncTransferOp>([&](auto op) {
          removeAssumedBits(NOT_TRANSFER_READ);
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.result()),
              DFX::Resolution::OPTIONAL);
          bool isSourceStaging =
              !(isAssumed(NOT_STAGING_READ) && isAssumed(NOT_STAGING_WRITE));
          bool isTargetStaging = !(resultUsage.isAssumed(NOT_STAGING_READ) &&
                                   resultUsage.isAssumed(NOT_STAGING_WRITE));
          if (isSourceStaging != isTargetStaging) {
            // Can't transition staging across transfers.
            LLVM_DEBUG({
              llvm::dbgs() << "[ValueResourceUsage] skipping transfer target: ";
              op.print(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << "\n";
            });
            return;
          }
          getState() ^= resultUsage.getState();
        })
        .Case<IREE::Stream::AsyncDispatchOp>([&](auto op) {
          removeAssumedBits(NOT_DISPATCH_READ);
          for (auto result : op.getOperandTiedResults(operandIdx)) {
            removeAssumedBits(NOT_MUTATED | NOT_DISPATCH_WRITE);
            auto resultUsage = solver.getElementFor<ValueResourceUsage>(
                *this, Position::forValue(result), DFX::Resolution::REQUIRED);
            getState() ^= resultUsage.getState();
          }
        })
        .Default([&](Operation *op) {});
  }

  ChangeStatus updateValue(Value value, DFX::Solver &solver) override {
    auto assumedBits = getAssumed();

    auto traversalResult = TraversalResult::COMPLETE;

    // Join with defining ops - of which there may be multiple if we come from
    // a branch/region argument or call result.
    traversalResult |=
        solver.getExplorer().walkDefiningOps(value, [&](OpResult result) {
          updateFromDefiningOp(value, result, solver);
          return WalkResult::advance();
        });

    // Join with using ops.
    traversalResult |=
        solver.getExplorer().walkTransitiveUses(value, [&](OpOperand &operand) {
          updateFromUse(value, operand, solver);
          return WalkResult::advance();
        });

    if (traversalResult == TraversalResult::INCOMPLETE) {
      removeAssumedBits(NOT_EXTERNAL);
    }
    return assumedBits == getAssumed() ? ChangeStatus::UNCHANGED
                                       : ChangeStatus::CHANGED;
  }

  friend class DFX::Solver;
};
const char ValueResourceUsage::ID = 0;

class UsageAnalysis {
 public:
  explicit UsageAnalysis(Operation *rootOp)
      : explorer(rootOp, TraversalAction::SHALLOW),
        solver(explorer, allocator) {
    explorer.setOpAction<mlir::FuncOp>(TraversalAction::RECURSE);
    explorer
        .setDialectAction<IREE::Stream::StreamDialect, IREE::Util::UtilDialect>(
            TraversalAction::RECURSE);
    explorer.initialize();
  }

  ResourceUsageBitfield lookupResourceUsage(Value value) {
    return tryLookupResourceUsage(value).getValueOr(
        ResourceUsageBitfield::Unknown);
  }

  llvm::Optional<ResourceUsageBitfield> tryLookupResourceUsage(Value value) {
    auto resourceUsage =
        solver.lookupElementFor<ValueResourceUsage>(Position::forValue(value));
    if (!resourceUsage) return llvm::None;
    return resourceUsage->getAssumedUsage();
  }

  LogicalResult run() {
    // TODO(benvanik): initialize globals and track usage through them.
    // Today we pin globals to <constant> or <variable> but it'd be nice to
    // set that based on actual usage here.
    //
    // Initialize globals that we need to resolve.
    // explorer.forEachGlobal([&](const auto *globalInfo) {
    //   auto globalType = globalInfo->op.type();
    //   if (globalType.template isa<IREE::Stream::ResourceType>()) {
    //     solver.getOrCreateElementFor<GlobalResourceUsage>(
    //         Position::forOperation(globalInfo->op));
    //   }
    // });

    // Initialize all SSA values we can do just with trivial search.
    explorer.walkValues([&](Value value) {
      if (value.getType().isa<IREE::Stream::ResourceType>()) {
        solver.getOrCreateElementFor<ValueResourceUsage>(
            Position::forValue(value));
      }
      return WalkResult::advance();
    });

    return solver.run();
  }

 private:
  Explorer explorer;
  llvm::BumpPtrAllocator allocator;
  DFX::Solver solver;
};

//===----------------------------------------------------------------------===//
// Resource usage query/application patterns
//===----------------------------------------------------------------------===//

template <typename Op>
struct UsageRefinementPattern : public OpRewritePattern<Op> {
  UsageRefinementPattern(MLIRContext *context, UsageAnalysis &analysis)
      : OpRewritePattern<Op>(context), analysis(analysis) {}

  UsageAnalysis &analysis;

  // Updates the |result| type to the lifetime derived by analysis, if needed.
  // Returns true if a change was made.
  bool applyResultTransition(Operation *op, Value result,
                             PatternRewriter &rewriter) const {
    auto oldType = result.getType().dyn_cast<IREE::Stream::ResourceType>();
    if (!oldType) return false;
    auto newUsage = analysis.lookupResourceUsage(result);
    auto newLifetime = convertUsageToLifetime(newUsage);
    if (oldType.getLifetime() == newLifetime) return false;
    auto newType =
        IREE::Stream::ResourceType::get(rewriter.getContext(), newLifetime);

    result.setType(newType);
    return true;
  }

  // Updates the |result| type to the lifetime derived by analysis, if needed.
  // Returns true if a change was made.
  bool applyResultTransition(Operation *op, Value result, Value resultSize,
                             Attribute affinityAttr,
                             PatternRewriter &rewriter) const {
    auto oldType = result.getType().dyn_cast<IREE::Stream::ResourceType>();
    if (!oldType) return false;
    auto newUsage = analysis.lookupResourceUsage(result);
    auto newLifetime = convertUsageToLifetime(newUsage);
    if (oldType.getLifetime() == newLifetime) return false;
    auto newType =
        IREE::Stream::ResourceType::get(rewriter.getContext(), newLifetime);

    result.setType(newType);
    return true;
  }

  bool applyRegionTransitions(Operation *op, PatternRewriter &rewriter) const {
    bool didChange = false;
    rewriter.startRootUpdate(op);
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        rewriter.setInsertionPoint(&block, block.begin());
        for (auto &blockArg : block.getArguments()) {
          auto oldType =
              blockArg.getType().dyn_cast<IREE::Stream::ResourceType>();
          if (!oldType) continue;
          auto newUsage = analysis.lookupResourceUsage(blockArg);
          auto newLifetime = convertUsageToLifetime(newUsage);
          if (oldType.getLifetime() == newLifetime) return false;
          auto newType = IREE::Stream::ResourceType::get(rewriter.getContext(),
                                                         newLifetime);
          blockArg.setType(newType);
          didChange |= true;
        }
      }
    }
    if (didChange) {
      rewriter.finalizeRootUpdate(op);
    } else {
      rewriter.cancelRootUpdate(op);
    }
    return didChange;
  }
};

struct ApplyInitializerOp
    : public UsageRefinementPattern<IREE::Util::InitializerOp> {
  using UsageRefinementPattern<
      IREE::Util::InitializerOp>::UsageRefinementPattern;
  LogicalResult matchAndRewrite(IREE::Util::InitializerOp op,
                                PatternRewriter &rewriter) const override {
    bool didChange = false;
    didChange = this->applyRegionTransitions(op, rewriter) || didChange;
    return didChange ? success() : failure();
  }
};

struct ApplyFuncOp : public UsageRefinementPattern<mlir::FuncOp> {
  using UsageRefinementPattern<mlir::FuncOp>::UsageRefinementPattern;
  LogicalResult matchAndRewrite(mlir::FuncOp op,
                                PatternRewriter &rewriter) const override {
    bool didChange = false;

    SmallVector<Type> newInputs;
    for (auto inputType : llvm::enumerate(op.getType().getInputs())) {
      auto oldType = inputType.value().dyn_cast<IREE::Stream::ResourceType>();
      if (!oldType) {
        newInputs.push_back(inputType.value());
        continue;
      }
      auto blockArg = op.getArgument(inputType.index());
      auto newUsage = analysis.lookupResourceUsage(blockArg);
      auto newLifetime = convertUsageToLifetime(newUsage);
      auto newType =
          IREE::Stream::ResourceType::get(rewriter.getContext(), newLifetime);
      newInputs.push_back(newType);
    }
    SmallVector<Type> newOutputs;
    auto anyReturnOp = *op.getOps<mlir::ReturnOp>().begin();
    for (auto outputType : llvm::enumerate(op.getType().getResults())) {
      auto oldType = outputType.value().dyn_cast<IREE::Stream::ResourceType>();
      if (!oldType) {
        newOutputs.push_back(outputType.value());
        continue;
      }
      auto returnValue = anyReturnOp.getOperand(outputType.index());
      auto newUsage = analysis.lookupResourceUsage(returnValue);
      auto newLifetime = convertUsageToLifetime(newUsage);
      auto newType =
          IREE::Stream::ResourceType::get(rewriter.getContext(), newLifetime);
      newOutputs.push_back(newType);
    }
    auto newFuncType = rewriter.getFunctionType(newInputs, newOutputs);
    if (op.getType() != newFuncType) {
      op.setType(newFuncType);
      didChange = true;
    }

    didChange = this->applyRegionTransitions(op, rewriter) || didChange;
    return didChange ? success() : failure();
  }
};

template <typename Op>
struct ApplyGenericOp : public UsageRefinementPattern<Op> {
  using UsageRefinementPattern<Op>::UsageRefinementPattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    bool didChange = this->applyRegionTransitions(op, rewriter);
    rewriter.startRootUpdate(op);
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      auto result = op->getResult(i);
      if (!result.getType().template isa<IREE::Stream::ResourceType>())
        continue;
      didChange = this->applyResultTransition(op, result, rewriter);
    }
    if (didChange) {
      rewriter.finalizeRootUpdate(op);
    } else {
      rewriter.cancelRootUpdate(op);
    }
    return didChange ? success() : failure();
  }
};

template <typename Op>
struct ApplyStreamableOp : public UsageRefinementPattern<Op> {
  using UsageRefinementPattern<Op>::UsageRefinementPattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    bool didChange = this->applyRegionTransitions(op, rewriter);
    Attribute affinityAttr;
    if (auto affinityOp =
            dyn_cast<IREE::Stream::AffinityOpInterface>(op.getOperation())) {
      affinityAttr = affinityOp.getAffinity();
    }

    rewriter.startRootUpdate(op);

    auto sizeAwareOp =
        dyn_cast<IREE::Util::SizeAwareOpInterface>(op.getOperation());
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      auto result = op->getResult(i);
      if (!result.getType().template isa<IREE::Stream::ResourceType>())
        continue;
      auto resultSize = sizeAwareOp.getResultSize(i);
      didChange = this->applyResultTransition(op, result, resultSize,
                                              affinityAttr, rewriter);
    }

    if (didChange) {
      rewriter.finalizeRootUpdate(op);
    } else {
      rewriter.cancelRootUpdate(op);
    }
    return didChange ? success() : failure();
  }
};

static void insertUsageRefinementPatterns(MLIRContext *context,
                                          UsageAnalysis &analysis,
                                          OwningRewritePatternList &patterns) {
  patterns.insert<ApplyInitializerOp, ApplyFuncOp>(context, analysis);
  patterns.insert<ApplyGenericOp<IREE::Util::DoNotOptimizeOp>,
                  ApplyGenericOp<mlir::CallOp>>(context, analysis);
  patterns.insert<ApplyStreamableOp<IREE::Stream::ResourceLoadOp>,
                  ApplyStreamableOp<IREE::Stream::ResourceStoreOp>,
                  ApplyStreamableOp<IREE::Stream::TensorImportOp>,
                  ApplyStreamableOp<IREE::Stream::TensorExportOp>,
                  ApplyStreamableOp<IREE::Stream::AsyncAllocaOp>,
                  ApplyStreamableOp<IREE::Stream::AsyncConstantOp>,
                  ApplyStreamableOp<IREE::Stream::AsyncSplatOp>,
                  ApplyStreamableOp<IREE::Stream::AsyncCloneOp>,
                  ApplyStreamableOp<IREE::Stream::AsyncSliceOp>,
                  ApplyStreamableOp<IREE::Stream::AsyncFillOp>,
                  ApplyStreamableOp<IREE::Stream::AsyncUpdateOp>,
                  ApplyStreamableOp<IREE::Stream::AsyncCopyOp>,
                  ApplyStreamableOp<IREE::Stream::AsyncTransferOp>,
                  ApplyStreamableOp<IREE::Stream::AsyncDispatchOp>,
                  ApplyStreamableOp<IREE::Stream::AsyncExecuteOp>,
                  ApplyStreamableOp<IREE::Stream::AsyncWaveOp>,
                  ApplyStreamableOp<IREE::Stream::YieldOp>>(context, analysis);
  IREE::Stream::AsyncTransferOp::getCanonicalizationPatterns(patterns, context);
}

//===----------------------------------------------------------------------===//
// -iree-stream-refine-usage
//===----------------------------------------------------------------------===//

class RefineUsagePass : public RefineUsageBase<RefineUsagePass> {
 public:
  RefineUsagePass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty()) return;

    // Run analysis on the entire module.
    UsageAnalysis analysis(moduleOp);
    if (failed(analysis.run())) {
      moduleOp.emitError() << "failed to solve for usage analysis";
      return signalPassFailure();
    }

    // Query and apply analysis results to all resources in the program.
    OwningRewritePatternList patterns(&getContext());
    insertUsageRefinementPatterns(&getContext(), analysis, patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(moduleOp, frozenPatterns))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createRefineUsagePass() {
  return std::make_unique<RefineUsagePass>();
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
