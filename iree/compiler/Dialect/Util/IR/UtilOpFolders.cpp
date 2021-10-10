// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

//===----------------------------------------------------------------------===//
// util.cmp.eq
//===----------------------------------------------------------------------===//

OpFoldResult CmpEQOp::fold(ArrayRef<Attribute> operands) {
  auto makeBool = [&](bool value) {
    return IntegerAttr::get(IntegerType::get(getContext(), 1), value ? 1 : 0);
  };
  if (lhs() == rhs()) {
    // SSA values are exactly the same.
    return makeBool(true);
  } else if (operands[0] && operands[1] && operands[0] == operands[1]) {
    // Folded attributes are equal but may come from separate ops.
    return makeBool(true);
  }
  // TODO(benvanik): we could add some interfaces for comparing, but this is
  // likely good enough for now.
  return {};
}

//===----------------------------------------------------------------------===//
// util.range.min/max
//===----------------------------------------------------------------------===//

static int64_t xmin(int64_t a, int64_t b) { return std::min(a, b); }
static int64_t xmax(int64_t a, int64_t b) { return std::max(a, b); }

template <int64_t initialValue, int64_t expr(int64_t, int64_t)>
static OpFoldResult foldRangeOp(Type type, ValueRange operands,
                                ArrayRef<Attribute> attrOperands) {
  // One operand is a pass-through.
  if (operands.size() == 1) {
    return operands.front();
  }

  // If all operands are constant then fold into a constant.
  int64_t value = initialValue;
  for (auto operand : attrOperands) {
    auto intValue = operand.dyn_cast_or_null<IntegerAttr>();
    if (!intValue) return {};
    value = expr(value, intValue.getValue().getSExtValue());
  }
  return IntegerAttr::get(type, value);
}

OpFoldResult RangeMinOp::fold(ArrayRef<Attribute> operands) {
  return foldRangeOp<INT64_MAX, xmin>(getType(), this->operands(), operands);
}

OpFoldResult RangeMaxOp::fold(ArrayRef<Attribute> operands) {
  return foldRangeOp<INT64_MIN, xmax>(getType(), this->operands(), operands);
}

namespace {

// Simplifies min/max ops by folding constants and deduplicating values.
//
// Example:
//  %min = util.range.min %0, %c1, %c2, %0, %1
// ->
//  %min = util.range.min %c1, %0, %1
template <typename OpT, int64_t initialValue, int64_t expr(int64_t, int64_t)>
struct SimplifyUniformRangeOp : public OpRewritePattern<OpT> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    SetVector<Value> operands;
    int64_t constantValue = initialValue;
    for (auto operand : op.operands()) {
      APInt constantInt;
      if (matchPattern(operand, m_ConstantInt(&constantInt))) {
        // Constant value.
        constantValue = expr(constantValue, constantInt.getSExtValue());
      } else {
        // Dynamic value.
        operands.insert(operand);
      }
    }
    if (operands.size() == op.operands().size()) return failure();
    if (constantValue != initialValue) {
      operands.insert(rewriter.create<mlir::ConstantIntOp>(
          op.getLoc(), constantValue, op.result().getType()));
    }
    rewriter.replaceOpWithNewOp<OpT>(op, op.result().getType(),
                                     operands.takeVector());
    return success();
  }
};

}  // namespace

void RangeMinOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<SimplifyUniformRangeOp<RangeMinOp, INT64_MAX, xmin> >(context);
}

void RangeMaxOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<SimplifyUniformRangeOp<RangeMaxOp, INT64_MIN, xmax> >(context);
}

//===----------------------------------------------------------------------===//
// util.range.extents
//===----------------------------------------------------------------------===//

static Value makeRangeEnd(Location loc, Value offset, Value length, Value one,
                          OpBuilder &builder) {
  return builder.create<mlir::SubIOp>(
      loc, builder.create<mlir::AddIOp>(loc, offset, length), one);
}
static Value makeRangeEnd(Location loc, Value offset, Value length,
                          OpBuilder &builder) {
  return makeRangeEnd(
      loc, offset, length,
      builder.create<mlir::ConstantIntOp>(loc, 1, offset.getType()), builder);
}

namespace {

struct FoldConstantRanges : public OpRewritePattern<RangeExtentsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RangeExtentsOp op,
                                PatternRewriter &rewriter) const override {
    // Build a constant range for all we find and preserve the dynamic pairs.
    SmallVector<Value> offsets;
    SmallVector<Value> lengths;
    offsets.reserve(op.offsets().size());
    lengths.reserve(op.lengths().size());
    int64_t constantMin = INT64_MAX;
    int64_t constantMax = INT64_MIN;
    for (auto range : llvm::zip(op.offsets(), op.lengths())) {
      auto offset = std::get<0>(range);
      auto length = std::get<1>(range);
      APInt rangeOffset, rangeLength;
      if (matchPattern(offset, m_ConstantInt(&rangeOffset)) &&
          matchPattern(length, m_ConstantInt(&rangeLength))) {
        // Both offset and length are constant so we can fold.
        constantMin = std::min(constantMin, rangeOffset.getSExtValue());
        constantMax = std::max(constantMax,
                               (rangeOffset + rangeLength - 1).getSExtValue());
      } else {
        // Dynamic value that we'll preserve.
        offsets.push_back(offset);
        lengths.push_back(length);
      }
    }
    if (offsets.size() == op.offsets().size()) return failure();

    // Preserve dynamic ranges.
    Value min;
    Value max;
    if (!offsets.empty()) {
      auto newOp =
          rewriter.create<RangeExtentsOp>(op.getLoc(), op.min().getType(),
                                          op.max().getType(), offsets, lengths);
      min = newOp.min();
      max = newOp.max();
    }

    // Min/max with constant ranges. This allows for normal folding to happen
    // downstream of the op.
    auto constantMinOp = rewriter.create<mlir::ConstantIntOp>(
        op.getLoc(), constantMin, op.min().getType());
    auto constantMaxOp = rewriter.create<mlir::ConstantIntOp>(
        op.getLoc(), constantMax - constantMin + 1, op.max().getType());
    min = min ? rewriter.create<mlir::MinUIOp>(op.getLoc(), min, constantMinOp)
                    .getResult()
              : constantMinOp.getResult();
    max = max ? rewriter.create<mlir::MaxUIOp>(op.getLoc(), max, constantMaxOp)
                    .getResult()
              : constantMaxOp.getResult();

    rewriter.replaceOp(op, {min, max});
    return success();
  }
};

struct ExpandSimpleRangeExtentsOp : public OpRewritePattern<RangeExtentsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RangeExtentsOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value minValue, maxValue;
    if (op.offsets().size() == 1) {
      // Single range folds to the min/max of that one range.
      minValue = op.offsets().front();
      maxValue = makeRangeEnd(loc, op.offsets().front(), op.lengths().front(),
                              rewriter);
    } else if (op.offsets().size() == 2) {
      // Two ranges turn into min/max.
      minValue = rewriter.create<mlir::MinUIOp>(loc, op.offsets().front(),
                                                op.offsets().back());
      auto one =
          rewriter.create<mlir::ConstantIntOp>(loc, 1, op.min().getType());
      auto endLhs = makeRangeEnd(loc, op.offsets().front(),
                                 op.lengths().front(), one, rewriter);
      auto endRhs = makeRangeEnd(loc, op.offsets().back(), op.lengths().back(),
                                 one, rewriter);
      maxValue = rewriter.create<mlir::MaxUIOp>(loc, endLhs, endRhs);
    }
    if (!minValue || !maxValue) return failure();
    rewriter.replaceOp(op, {minValue, maxValue});
    return success();
  }
};

// Simplifies cases where the extents can be expressed as min/max.
//
// Examples:
//  %min, %max = util.range.extents [%o for %l0], [%o for %l1]
// ->
//  %min = %o
//  %max_len = util.range.max %l0, %l1
//  %max = addi %min, %max_len
struct SimplifyUniformRangeExtentsOp : public OpRewritePattern<RangeExtentsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RangeExtentsOp op,
                                PatternRewriter &rewriter) const override {
    // First filter out any pure duplicates. Note SetVector so order is
    // preserved.
    using Range = std::tuple<Value, Value>;
    SetVector<Range> ranges;
    for (auto range : llvm::zip(op.offsets(), op.lengths())) {
      ranges.insert(range);
    }

    // Build equivalence classes of all the ranges that share the same offsets
    // or same lengths.
    llvm::EquivalenceClasses<unsigned> offsetsEC;
    llvm::EquivalenceClasses<unsigned> lengthsEC;
    DenseMap<Value, unsigned> offsetMap;  // SSA value -> base class
    DenseMap<Value, unsigned> lengthMap;  // SSA value -> base class
    for (auto range : llvm::enumerate(ranges)) {
      auto offset = std::get<0>(range.value());
      auto length = std::get<1>(range.value());
      auto offsetIt = offsetMap.find(offset);
      if (offsetIt == offsetMap.end()) {
        offsetMap.insert({offset, range.index()});  // first seen
        offsetsEC.insert(range.index());
      } else {
        offsetsEC.unionSets(offsetIt->second, range.index());
      }
      auto lengthIt = lengthMap.find(length);
      if (lengthIt == lengthMap.end()) {
        lengthMap.insert({length, range.index()});  // first seen
        lengthsEC.insert(range.index());
      } else {
        lengthsEC.unionSets(lengthIt->second, range.index());
      }
    }

    size_t oldRangeCount = op.offsets().size();
    if (ranges.size() == oldRangeCount &&
        offsetsEC.getNumClasses() == oldRangeCount &&
        lengthsEC.getNumClasses() == oldRangeCount) {
      // No duplicates or equivalent offsets/lengths.
      return failure();
    }

    // Handle the equivalent offsets/lengths; note we do this after removing
    // dupes so that we know we can process each independently (all ranges are
    // going to be same offset, same length, or neither).
    llvm::BitVector handledRanges;
    handledRanges.resize(ranges.size(), /*t=*/false);
    SmallVector<Value> minOffsets;
    SmallVector<Value> maxLengths;
    for (auto it = offsetsEC.begin(), end = offsetsEC.end(); it != end; ++it) {
      if (!it->isLeader()) continue;  // Ignore non-leader sets.
      if (++offsetsEC.member_begin(it) == offsetsEC.member_end()) continue;
      // Shared offsets -> max on lengths.
      minOffsets.push_back(std::get<0>(ranges[it->getData()]));
      for (auto mi = offsetsEC.member_begin(it); mi != offsetsEC.member_end();
           ++mi) {
        maxLengths.push_back(std::get<1>(ranges[*mi]));
      }
    }
    for (auto it = lengthsEC.begin(), end = lengthsEC.end(); it != end; ++it) {
      if (!it->isLeader()) continue;  // Ignore non-leader sets.
      if (++lengthsEC.member_begin(it) == lengthsEC.member_end()) continue;
      // Shared lengths -> min on offsets.
      maxLengths.push_back(std::get<1>(ranges[it->getData()]));
      for (auto mi = lengthsEC.member_begin(it); mi != lengthsEC.member_end();
           ++mi) {
        minOffsets.push_back(std::get<0>(ranges[*mi]));
      }
    }

    // Preserve any independent offset/length ranges in a new extents op.
    SmallVector<Value> offsets;
    SmallVector<Value> lengths;
    offsets.reserve(ranges.size());
    lengths.reserve(ranges.size());
    for (auto range : llvm::enumerate(ranges)) {
      if (handledRanges.test(range.index())) continue;
      offsets.push_back(std::get<0>(range.value()));
      lengths.push_back(std::get<1>(range.value()));
    }
    Value min;
    Value max;
    if (offsets.empty()) {
      auto newOp =
          rewriter.create<RangeExtentsOp>(op.getLoc(), op.min().getType(),
                                          op.max().getType(), offsets, lengths);
      min = newOp.min();
      max = newOp.max();
    }

    // Handle offset mins.
    if (!minOffsets.empty()) {
      min = rewriter
                .create<RangeMinOp>(op.getLoc(), op.min().getType(), minOffsets)
                .result();
    }

    // Handle offset maxs.
    if (!maxLengths.empty()) {
      auto maxLength =
          rewriter
              .create<RangeMaxOp>(op.getLoc(), op.max().getType(), maxLengths)
              .result();
      auto maxEnd = rewriter.create<mlir::AddIOp>(op.getLoc(), min, maxLength);
      max = rewriter.create<mlir::MaxUIOp>(op.getLoc(), max, maxEnd);
    }

    rewriter.replaceOp(op, {min, max});
    return success();
  }
};

}  // namespace

void RangeExtentsOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<FoldConstantRanges>(context);
  results.insert<ExpandSimpleRangeExtentsOp>(context);
  results.insert<SimplifyUniformRangeExtentsOp>(context);
}

//===----------------------------------------------------------------------===//
// Compiler hints
//===----------------------------------------------------------------------===//

namespace {

struct ExpandUnfoldableConstantOp
    : public OpRewritePattern<UnfoldableConstantOp> {
  using OpRewritePattern<IREE::Util::UnfoldableConstantOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(UnfoldableConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto stdConst = rewriter.create<ConstantOp>(op.getLoc(), op.value());
    rewriter.replaceOpWithNewOp<DoNotOptimizeOp>(op, stdConst.getResult());
    return success();
  }
};

}  // namespace

void UnfoldableConstantOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ExpandUnfoldableConstantOp>(context);
}

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

namespace {

// Deletes empty vm.initializer ops.
struct DropEmptyInitializerOp : public OpRewritePattern<InitializerOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InitializerOp op,
                                PatternRewriter &rewriter) const override {
    if (op.body().getBlocks().size() != 1) return failure();
    auto &block = op.body().front();
    if (block.empty() || isa<InitializerReturnOp>(block.front())) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

// Inlines constant stores from initializers into the global initializer.
// This is not strictly required but can help our initialization code perform
// more efficient initialization of large numbers of primitive values.
struct InlineConstantGlobalInitializer
    : public OpRewritePattern<InitializerOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InitializerOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Operation *> deadOps;
    op.walk([&](GlobalStoreOp storeOp) {
      Attribute valueAttr;
      if (!matchPattern(storeOp.value(), m_Constant(&valueAttr))) return;
      auto globalOp = storeOp.getGlobalOp();
      rewriter.updateRootInPlace(globalOp, [&]() {
        if (valueAttr && !valueAttr.isa<UnitAttr>()) {
          globalOp.initial_valueAttr(valueAttr);
        } else {
          globalOp.clearInitialValue();
        }
      });
      deadOps.push_back(storeOp);
    });
    if (deadOps.empty()) return failure();
    for (auto deadOp : deadOps) rewriter.eraseOp(deadOp);
    return success();
  }
};

}  // namespace

void InitializerOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<DropEmptyInitializerOp, InlineConstantGlobalInitializer>(
      context);
}

void GlobalOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {}

namespace {

/// Turns util.global.address -> util.global.load.indirect into a direct load.
class PropagateGlobalLoadAddress
    : public OpRewritePattern<GlobalLoadIndirectOp> {
  using OpRewritePattern::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(GlobalLoadIndirectOp op,
                                PatternRewriter &rewriter) const override {
    if (auto addressOp =
            dyn_cast_or_null<GlobalAddressOp>(op.global().getDefiningOp())) {
      rewriter.replaceOpWithNewOp<GlobalLoadOp>(op, op.result().getType(),
                                                addressOp.global());
      return success();
    }
    return failure();
  }
};

}  // namespace

void GlobalLoadIndirectOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<PropagateGlobalLoadAddress>(context);
}

namespace {

/// Erases util.global.store ops that are no-ops.
/// This can happen if there was a global load, some DCE'd usage, and a
/// store back to the same global: we want to be able to elide the entire load
/// and store.
struct EraseUnusedGlobalStoreOp : public OpRewritePattern<GlobalStoreOp> {
  using OpRewritePattern<GlobalStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GlobalStoreOp op,
                                PatternRewriter &rewriter) const override {
    if (auto loadOp =
            dyn_cast_or_null<GlobalLoadOp>(op.value().getDefiningOp())) {
      if (loadOp.global() == op.global()) {
        rewriter.eraseOp(op);
        return success();
      }
    }
    return failure();
  }
};

}  // namespace

void GlobalStoreOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<EraseUnusedGlobalStoreOp>(context);
}

namespace {

/// Turns util.global.address -> util.global.store.indirect into a direct store.
class PropagateGlobalStoreAddress
    : public OpRewritePattern<GlobalStoreIndirectOp> {
  using OpRewritePattern::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(GlobalStoreIndirectOp op,
                                PatternRewriter &rewriter) const override {
    if (auto addressOp =
            dyn_cast_or_null<GlobalAddressOp>(op.global().getDefiningOp())) {
      rewriter.replaceOpWithNewOp<GlobalStoreOp>(op, op.value(),
                                                 addressOp.global());
      return success();
    }
    return failure();
  }
};

}  // namespace

void GlobalStoreIndirectOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<PropagateGlobalStoreAddress>(context);
}

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
