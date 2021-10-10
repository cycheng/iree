// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "llvm/ADT/BitVector.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {
namespace {

struct DescriptorState {
  Value buffer;
  Value offset;
  Value length;
};

struct DescriptorSetState {
  Value executableLayout;
  SmallVector<DescriptorState, 32> descriptors;

  DescriptorState &getDescriptor(int64_t index) {
    if (index >= descriptors.size()) {
      descriptors.resize(index + 1);
    }
    return descriptors[index];
  }

  void clear() {
    executableLayout = {};
    descriptors.clear();
  }
};

struct CommandBufferState {
  SmallVector<Value, 32> pushConstants;
  SmallVector<DescriptorSetState, 4> descriptorSets;

  Value &getPushConstant(int64_t index) {
    if (index >= pushConstants.size()) {
      pushConstants.resize(index + 1);
    }
    return pushConstants[index];
  }

  DescriptorSetState *getDescriptorSet(Value set) {
    APInt setInt;
    if (!matchPattern(set, m_ConstantInt(&setInt))) {
      // Dynamic set value; not analyzable with this approach.
      return nullptr;
    }
    int64_t index = setInt.getSExtValue();
    if (index >= descriptorSets.size()) {
      descriptorSets.resize(index + 1);
    }
    return &descriptorSets[index];
  }
};

using CommandBufferStateMap = DenseMap<Value, CommandBufferState>;

}  // namespace

static LogicalResult processOp(IREE::HAL::CommandBufferPushConstantsOp op,
                               CommandBufferState &state) {
  // Today we only eat constants from the beginning or end of the range
  // (hopefully removing the entire op). Sparse constant sets aren't worth it.
  int64_t baseIndex = op.offset().getSExtValue();
  llvm::BitVector redundantIndices(baseIndex + op.values().size());
  for (auto value : llvm::enumerate(op.values())) {
    auto &stateValue = state.getPushConstant(baseIndex + value.index());
    if (value.value() == stateValue) {
      // Redundant value.
      redundantIndices.set(value.index());
    } else {
      stateValue = value.value();
    }
  }
  if (redundantIndices.none()) return success();  // no-op

  // If all bits are set we can just kill the op.
  if (redundantIndices.all()) {
    op.erase();
    return success();
  }

  int lastRedundant = redundantIndices.find_last();
  if (lastRedundant != -1 && lastRedundant != redundantIndices.size()) {
    // Eat the last few constants.
    int redundantCount = redundantIndices.size() - lastRedundant;
    op.valuesMutable().erase(lastRedundant, redundantCount);
  }

  int firstRedundant = redundantIndices.find_first();
  if (firstRedundant != -1 && firstRedundant != 0) {
    // Eat the first few constants by adjusting the offset and slicing out the
    // values.
    op.offsetAttr(Builder(op).getIndexAttr(baseIndex + firstRedundant));
    op.valuesMutable().erase(0, firstRedundant);
  }

  return success();
}

static LogicalResult processOp(IREE::HAL::CommandBufferPushDescriptorSetOp op,
                               CommandBufferState &state) {
  auto *setState = state.getDescriptorSet(op.set());
  if (!setState) return failure();

  bool isLayoutEqual = setState->executableLayout == op.executable_layout();
  setState->executableLayout = op.executable_layout();

  int64_t descriptorCount = op.binding_buffers().size();
  llvm::BitVector redundantIndices(descriptorCount);
  for (int64_t index = 0; index < descriptorCount; ++index) {
    auto &descriptor = setState->getDescriptor(index);
    auto buffer = op.binding_buffers()[index];
    auto offset = op.binding_offsets()[index];
    auto length = op.binding_lengths()[index];
    if (descriptor.buffer == buffer && descriptor.offset == offset &&
        descriptor.length == length) {
      // Redundant descriptor.
      redundantIndices.set(index);
    } else {
      descriptor.buffer = buffer;
      descriptor.offset = offset;
      descriptor.length = length;
    }
  }

  // Bail early if no redundant bindings.
  if (isLayoutEqual && redundantIndices.none()) {
    return success();  // no-op
  }

  // If all bits are set we can just kill the op.
  if (isLayoutEqual && redundantIndices.all()) {
    op.erase();
    return success();
  }

  return success();
}

static LogicalResult processOp(IREE::HAL::CommandBufferBindDescriptorSetOp op,
                               CommandBufferState &state) {
  // TODO(benvanik): descriptor set binding.
  // For now we just nuke the state.
  auto *setState = state.getDescriptorSet(op.set());
  if (!setState) return failure();
  setState->clear();
  return success();
}

class ElideRedundantCommandsPass
    : public PassWrapper<ElideRedundantCommandsPass, OperationPass<void>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
  }

  StringRef getArgument() const override {
    return "iree-hal-elide-redundant-commands";
  }

  StringRef getDescription() const override {
    return "Elides stateful command buffer ops that set redundant state.";
  }

  void runOnOperation() override {
    auto parentOp = getOperation();

    // TODO(benvanik): intraprocedural at least; keep track of state at block
    // boundaries. IPO would be nice but it (today) rarely happens that we
    // pass command buffers across calls.
    for (auto &region : parentOp->getRegions()) {
      for (auto &block : region.getBlocks()) {
        // State tracking for each command buffer found.
        // Discard state on ops we don't currently analyze (because this is
        // super basic - we really need to analyze them).
        CommandBufferStateMap stateMap;
        auto invalidateState = [&](Value commandBuffer) {
          stateMap[commandBuffer] = {};
        };
        for (auto &op : llvm::make_early_inc_range(block.getOperations())) {
          if (op.getDialect())
            TypeSwitch<Operation *>(&op)
                .Case([&](IREE::HAL::CommandBufferBeginOp op) {
                  invalidateState(op.command_buffer());
                })
                .Case([&](IREE::HAL::CommandBufferEndOp op) {
                  invalidateState(op.command_buffer());
                })
                .Case([&](IREE::HAL::CommandBufferPushConstantsOp op) {
                  if (failed(processOp(op, stateMap[op.command_buffer()]))) {
                    invalidateState(op.command_buffer());
                  }
                })
                .Case([&](IREE::HAL::CommandBufferPushDescriptorSetOp op) {
                  if (failed(processOp(op, stateMap[op.command_buffer()]))) {
                    invalidateState(op.command_buffer());
                  }
                })
                .Case([&](IREE::HAL::CommandBufferBindDescriptorSetOp op) {
                  if (failed(processOp(op, stateMap[op.command_buffer()]))) {
                    invalidateState(op.command_buffer());
                  }
                })
                .Case<IREE::HAL::CommandBufferDeviceOp,
                      IREE::HAL::CommandBufferBeginDebugGroupOp,
                      IREE::HAL::CommandBufferEndDebugGroupOp,
                      IREE::HAL::CommandBufferExecutionBarrierOp,
                      IREE::HAL::CommandBufferFillBufferOp,
                      IREE::HAL::CommandBufferCopyBufferOp,
                      IREE::HAL::CommandBufferDispatchSymbolOp,
                      IREE::HAL::CommandBufferDispatchOp,
                      IREE::HAL::CommandBufferDispatchIndirectSymbolOp,
                      IREE::HAL::CommandBufferDispatchIndirectOp>(
                    [&](Operation *op) {
                      // Ok - don't impact state.
                    })
                .Default([&](Operation *op) {
                  // Unknown op - discard state cache.
                  // This is to avoid correctness issues with region ops (like
                  // scf.if) that we don't analyze properly here. We could
                  // restrict this a bit by only discarding on use of the
                  // command buffer.
                  stateMap.clear();
                });
        }
      }
    }
  }
};

std::unique_ptr<OperationPass<void>> createElideRedundantCommandsPass() {
  return std::make_unique<ElideRedundantCommandsPass>();
}

static PassRegistration<ElideRedundantCommandsPass> pass;

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
