// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion2/StandardToHAL/ConvertStandardToHAL.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "iree/compiler/Dialect/Util/Conversion/ConversionPatterns.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct ConvertTensorCastOp
    : public OpConversionPattern<IREE::HAL::TensorCastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::TensorCastOp op, llvm::ArrayRef<Value> rawOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::HAL::TensorCastOpAdaptor newOperands(
        rawOperands, op.getOperation()->getAttrDictionary());
    Value newValue = {};
    auto targetType = op.target().getType();
    if (targetType.isa<TensorType>()) {
      // HAL type -> tensor<...>
      newValue = newOperands.source();
    } else if (targetType.isa<IREE::HAL::BufferType>()) {
      // tensor<...> -> !hal.buffer
      auto adaptor = IREE::HAL::TensorRewriteAdaptor::get(
          op.getLoc(), op.source(), newOperands.source(), rewriter);
      newValue = adaptor.getBuffer();
    } else if (targetType.isa<IREE::HAL::BufferViewType>()) {
      // tensor<...> -> !hal.buffer_view
      auto adaptor = IREE::HAL::TensorRewriteAdaptor::get(
          op.getLoc(), op.source(), newOperands.source(), rewriter);

      // Note that the buffer view cannot just be returned here: it's backing
      // buffer will be correct, but the cast may be doing a metadata change,
      // which must be reflected in the returned buffer view. For now, we
      // just create a new view unconditionally when converting from a tensor
      // since that is conservative. But this can be optimized with additional
      // heuristics regarding when it is safe to alias the original.
      Value originalValue = op.source();
      if (auto sourceType =
              originalValue.getType().dyn_cast<RankedTensorType>()) {
        auto shapeDims = getShapeDims(rewriter, op.getLoc(), sourceType,
                                      newOperands.source_dims());
        newValue = rewriter.create<IREE::HAL::BufferViewCreateOp>(
            op.getLoc(), adaptor.getBuffer(), adaptor.getElementType(),
            adaptor.getEncodingType(), shapeDims);
      } else {
        newValue = adaptor.getBufferView();
      }
    }
    if (!newValue) {
      return rewriter.notifyMatchFailure(op, "bad source/target type pair");
    }
    rewriter.replaceOp(op, {newValue});
    return success();
  }

  SmallVector<Value> getShapeDims(OpBuilder &builder, Location loc,
                                  RankedTensorType sourceType,
                                  ValueRange sourceDims) const {
    SmallVector<Value> shapeDims(sourceType.getRank());
    int sourceDimsIndex = 0;
    for (int i = 0, e = shapeDims.size(); i < e; ++i) {
      if (sourceType.isDynamicDim(i)) {
        shapeDims[i] = sourceDims[sourceDimsIndex++];
      } else {
        shapeDims[i] =
            builder.create<ConstantIndexOp>(loc, sourceType.getDimSize(i));
      }
    }
    return shapeDims;
  }
};

struct ConvertIfOp : public OpConversionPattern<scf::IfOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      scf::IfOp ifOp, llvm::ArrayRef<Value> rawOperands,
      ConversionPatternRewriter &rewriter) const override {
    scf::IfOp::Adaptor operands(rawOperands);
    auto resultTypes = llvm::to_vector<4>(llvm::map_range(
        ifOp.getResultTypes(),
        [&](Type type) { return getTypeConverter()->convertType(type); }));
    auto newOp = rewriter.create<scf::IfOp>(ifOp.getLoc(), resultTypes,
                                            operands.condition(),
                                            ifOp.elseBlock() != nullptr);
    rewriter.inlineRegionBefore(ifOp.thenRegion(), newOp.thenRegion(),
                                newOp.thenRegion().end());
    rewriter.eraseBlock(&newOp.thenRegion().front());
    if (ifOp.elseBlock()) {
      rewriter.inlineRegionBefore(ifOp.elseRegion(), newOp.elseRegion(),
                                  newOp.elseRegion().end());
      rewriter.eraseBlock(&newOp.elseRegion().front());
    }
    rewriter.replaceOp(ifOp, newOp.results());
    return success();
  }
};

struct ConvertYieldOp : public OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      scf::YieldOp yieldOp, llvm::ArrayRef<Value> rawOperands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, rawOperands);
    return success();
  }
};

}  // namespace

void populateStandardShapeToHALPatterns(MLIRContext *context,
                                        OwningRewritePatternList &patterns,
                                        TypeConverter &converter);

void populateStandardStructuralToHALPatterns(MLIRContext *context,
                                             OwningRewritePatternList &patterns,
                                             TypeConverter &converter);

void populateStandardToHALPatterns(MLIRContext *context,
                                   ConversionTarget &conversionTarget,
                                   TypeConverter &typeConverter,
                                   OwningRewritePatternList &patterns) {
  conversionTarget.addLegalOp<mlir::ModuleOp>();

  // We need to rewrite certain types on operands/results so use the default
  // dynamic legality checker to force any ops using such types to run through
  // our patterns.
  conversionTarget.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getType()) &&
           typeConverter.isLegal(&op.getBody());
  });

  // Ensure all shape related ops are fully converted as we should no longer
  // have any types they are valid to be used on after this conversion.
  conversionTarget.addIllegalOp<mlir::RankOp>();
  conversionTarget.addIllegalOp<tensor::DimOp>();

  populateStandardShapeToHALPatterns(context, patterns, typeConverter);
  populateStandardStructuralToHALPatterns(context, patterns, typeConverter);

  // TODO(benvanik): move to general utils conversion.
  addGenericLegalOp<scf::IfOp>(conversionTarget, typeConverter);
  addGenericLegalOp<scf::YieldOp>(conversionTarget, typeConverter);
  patterns.insert<ConvertIfOp, ConvertYieldOp>(typeConverter, context);

  // TODO(benvanik): move to a ConvertTensorOps.cpp.
  conversionTarget.addIllegalOp<IREE::HAL::TensorCastOp>();
  patterns.insert<ConvertTensorCastOp>(context);
}

}  // namespace iree_compiler
}  // namespace mlir
