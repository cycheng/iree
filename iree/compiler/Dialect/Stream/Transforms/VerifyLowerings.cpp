// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {

//===----------------------------------------------------------------------===//
// Base pass utility
//===----------------------------------------------------------------------===//

class Verifier {
 public:
  enum class Legality {
    LEGAL,
    RECURSIVELY_LEGAL,
    ILLEGAL,
  };

  using OpVerifierFn = std::function<Optional<Legality>(Operation *op)>;
  using TypeVerifierFn = std::function<Legality(Type type)>;

  void addIllegalDialect(StringRef dialectName) {
    dialectLegality.insert({dialectName, Legality::ILLEGAL});
  }
  template <typename DialectT>
  void addIllegalDialect() {
    addIllegalDialect(DialectT::getDialectNamespace());
  }

  template <typename OpT>
  void addLegalOp() {
    opLegality.insert({OpT::getOperationName(), Legality::LEGAL});
  }

  template <typename OpT>
  void addRecursivelyLegalOp() {
    opLegality.insert({OpT::getOperationName(), Legality::RECURSIVELY_LEGAL});
  }

  template <typename OpT>
  void addIllegalOp() {
    opLegality.insert({OpT::getOperationName(), Legality::ILLEGAL});
  }

  template <typename OpT>
  void addOpVerifier(std::function<Optional<Legality>(OpT)> fn) {
    auto wrapperFn = [=](Operation *baseOp) -> Optional<Legality> {
      if (auto op = dyn_cast<OpT>(baseOp)) {
        return fn(op);
      }
      return llvm::None;
    };
    opVerifiers.push_back(wrapperFn);
  }

  template <typename TypeT>
  void addIllegalType() {
    typeLegality.insert({TypeID::get<TypeT>(), Legality::ILLEGAL});
  }

  template <typename TypeT>
  void addTypeVerifier(std::function<Legality(TypeT)> fn) {
    auto wrapperFn = [=](Type baseType) { return fn(baseType.cast<TypeT>()); };
    if (typeVerifiers.insert({TypeID::get<TypeT>(), wrapperFn}).second ==
        false) {
      llvm_unreachable("already registered for this type");
    }
  }

  LogicalResult run(Operation *rootOp) {
    bool foundAnyIllegal = false;
    rootOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      auto walkResult = WalkResult::advance();

      // Check for op legality - can skip the expensive work if known-illegal.
      auto legality = getOpLegality(op);
      switch (legality) {
        case Legality::LEGAL:
          // Op itself is legal but may not have valid operands/results.
          break;
        case Legality::RECURSIVELY_LEGAL:
          // If the entire op w/ nested ops is legal then skip.
          return WalkResult::skip();
        default:
        case Legality::ILLEGAL:
          // Early-exit on illegal ops without recursing.
          emitIllegalOpError(op);
          foundAnyIllegal = true;
          return WalkResult::skip();
      }

      // Check types for operands/results.
      for (auto operandType : llvm::enumerate(op->getOperandTypes())) {
        if (isTypeLegal(operandType.value())) continue;
        emitIllegalTypeError(op, "operand", operandType.index(),
                             operandType.value());
        foundAnyIllegal = true;
      }
      for (auto resultType : llvm::enumerate(op->getResultTypes())) {
        if (isTypeLegal(resultType.value())) continue;
        emitIllegalTypeError(op, "result", resultType.index(),
                             resultType.value());
        foundAnyIllegal = true;
      }

      return walkResult;
    });
    return success(!foundAnyIllegal);
  }

 private:
  Legality getOpLegality(Operation *op) {
    auto opName = op->getName();

    // Check specific ops first (we may override dialect settings).
    {
      auto legalityIt = opLegality.find(opName.getStringRef());
      if (legalityIt != opLegality.end()) {
        return legalityIt->second;
      }
    }

    // Check all op verifiers (usually used for interface checks).
    for (auto &opVerifier : opVerifiers) {
      auto legalOr = opVerifier(op);
      if (legalOr.hasValue()) {
        return legalOr.getValue();
      }
    }

    // If no op carveout is applied then check to see if the dialect is
    // allowed at all.
    {
      auto legalityIt = dialectLegality.find(opName.getDialectNamespace());
      if (legalityIt != dialectLegality.end()) {
        return legalityIt->second;
      }
    }

    // Assume legal by default.
    return Legality::LEGAL;
  }

  bool isTypeLegal(Type type) {
    // TODO(benvanik): subelements interface checks using recursive legality.

    // Defer to verifiers first.
    auto it = typeVerifiers.find(type.getTypeID());
    if (it != typeVerifiers.end()) {
      return it->second(type) != Legality::ILLEGAL;
    }

    // Check legality of the base type.
    {
      auto legalityIt = typeLegality.find(type.getTypeID());
      if (legalityIt != typeLegality.end()) {
        return legalityIt->second != Legality::ILLEGAL;
      }
    }

    // Assume legal by default.
    return true;
  }

  void emitIllegalOpError(Operation *op) {
    op->emitOpError()
        << "illegal for this phase of lowering in the stream dialect; "
           "expected to have been converted or removed";
  }

  void emitIllegalTypeError(Operation *op, StringRef location, unsigned idx,
                            Type type) {
    op->emitOpError()
        << location << " " << idx << " type " << type
        << " illegal for this phase of lowering in the stream dialect";
  }

  DenseMap<StringRef, Legality> dialectLegality;
  DenseMap<StringRef, Legality> opLegality;
  SmallVector<OpVerifierFn> opVerifiers;
  DenseMap<TypeID, Legality> typeLegality;
  DenseMap<TypeID, TypeVerifierFn> typeVerifiers;
};

//===----------------------------------------------------------------------===//
// -iree-stream-verify-lowering-to-tensors
//===----------------------------------------------------------------------===//

static void markTensorInputsIllegal(Verifier &verifier) {
  // Tensorish dialects should all be either converted or outlined into
  // executables. Everything should be in resources now.
  verifier.addIllegalDialect("tensor");
  verifier.addIllegalDialect("linalg");

  // We don't allow the flow dialect except for inside of executables for which
  // we don't yet have a full mapping to in the stream dialect.
  // TODO(#7277): remove this carveout once we switch over to streams fully.
  verifier.addIllegalDialect("flow");
  verifier.addRecursivelyLegalOp<IREE::Stream::ExecutableOp>();
}

namespace {

class VerifyLoweringToTensorsPass
    : public VerifyLoweringToTensorsBase<VerifyLoweringToTensorsPass> {
 public:
  VerifyLoweringToTensorsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    Verifier verifier;
    markTensorInputsIllegal(verifier);
    if (failed(verifier.run(getOperation()))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createVerifyLoweringToTensorsPass() {
  return std::make_unique<VerifyLoweringToTensorsPass>();
}

//===----------------------------------------------------------------------===//
// -iree-stream-verify-lowering-to-tensors
//===----------------------------------------------------------------------===//

static void markStreamTensorOpsIllegal(Verifier &verifier) {
  // No more stream.tensor.* ops are allowed - all should be converted to
  // stream.async.*.
  // TODO(benvanik): traits to make this simpler.
  verifier.addIllegalOp<IREE::Stream::TensorCloneOp>();
  verifier.addIllegalOp<IREE::Stream::TensorConstantOp>();
  verifier.addIllegalOp<IREE::Stream::TensorFillOp>();
  verifier.addIllegalOp<IREE::Stream::TensorLoadOp>();
  verifier.addIllegalOp<IREE::Stream::TensorSizeOfOp>();
  verifier.addIllegalOp<IREE::Stream::TensorSliceOp>();
  verifier.addIllegalOp<IREE::Stream::TensorSplatOp>();
  verifier.addIllegalOp<IREE::Stream::TensorStoreOp>();
  verifier.addIllegalOp<IREE::Stream::TensorUpdateOp>();
}

namespace {

class VerifyLoweringToAsyncPass
    : public VerifyLoweringToAsyncBase<VerifyLoweringToAsyncPass> {
 public:
  VerifyLoweringToAsyncPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    Verifier verifier;
    markTensorInputsIllegal(verifier);
    markStreamTensorOpsIllegal(verifier);

    // All resources should have had their usage assigned.
    verifier.addTypeVerifier<IREE::Stream::ResourceType>([](auto type) {
      if (type.getLifetime() == IREE::Stream::Lifetime::Unknown) {
        return Verifier::Legality::ILLEGAL;
      }
      return Verifier::Legality::LEGAL;
    });

    // All streamable ops should be inside of execution regions.
    verifier.addOpVerifier<StreamableOpInterface>(
        [](auto op) -> Optional<Verifier::Legality> {
          // Allow metadata ops outside of execution regions.
          if (op.isMetadata()) return Verifier::Legality::LEGAL;

          // TODO(benvanik): execution region interface to make this generic.
          if (!op->getParentOfType<IREE::Stream::AsyncExecuteOp>()) {
            op->emitOpError()
                << ": streamable op expected to be in an execution region";
            return Verifier::Legality::ILLEGAL;
          }
          return llvm::None;
        });

    if (failed(verifier.run(getOperation()))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createVerifyLoweringToAsyncPass() {
  return std::make_unique<VerifyLoweringToAsyncPass>();
}

//===----------------------------------------------------------------------===//
// -iree-stream-verify-lowering-to-cmd
//===----------------------------------------------------------------------===//

static void markStreamAsyncOpsIllegal(Verifier &verifier) {
  // No more  stream.async.* ops are allowed - all should be converted to
  // stream.cmd.*.
  // TODO(benvanik): traits to make this simpler.
  verifier.addIllegalOp<IREE::Stream::AsyncAllocaOp>();
  verifier.addIllegalOp<IREE::Stream::AsyncCloneOp>();
  verifier.addIllegalOp<IREE::Stream::AsyncConstantOp>();
  verifier.addIllegalOp<IREE::Stream::AsyncCopyOp>();
  verifier.addIllegalOp<IREE::Stream::AsyncDispatchOp>();
  verifier.addIllegalOp<IREE::Stream::AsyncFillOp>();
  verifier.addIllegalOp<IREE::Stream::AsyncSliceOp>();
  verifier.addIllegalOp<IREE::Stream::AsyncSplatOp>();
  verifier.addIllegalOp<IREE::Stream::AsyncTransferOp>();
  verifier.addIllegalOp<IREE::Stream::AsyncUpdateOp>();
  verifier.addIllegalOp<IREE::Stream::AsyncExecuteOp>();
  verifier.addIllegalOp<IREE::Stream::AsyncWaveOp>();
}

namespace {

class VerifyLoweringToCmdPass
    : public VerifyLoweringToCmdBase<VerifyLoweringToCmdPass> {
 public:
  VerifyLoweringToCmdPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    Verifier verifier;
    markTensorInputsIllegal(verifier);
    markStreamTensorOpsIllegal(verifier);
    markStreamAsyncOpsIllegal(verifier);

    // All resources should have had their usage assigned.
    verifier.addTypeVerifier<IREE::Stream::ResourceType>([](auto type) {
      if (type.getLifetime() == IREE::Stream::Lifetime::Unknown) {
        return Verifier::Legality::ILLEGAL;
      }
      return Verifier::Legality::LEGAL;
    });

    if (failed(verifier.run(getOperation()))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createVerifyLoweringToCmdPass() {
  return std::make_unique<VerifyLoweringToCmdPass>();
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
