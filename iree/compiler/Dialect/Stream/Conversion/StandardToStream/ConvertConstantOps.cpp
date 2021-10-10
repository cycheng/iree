// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Conversion/StandardToStream/ConvertStandardToStream.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class ConvertTensorConstantOp : public OpConversionPattern<mlir::ConstantOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::ConstantOp constantOp, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    if (!constantOp.getType().isa<TensorType>()) return failure();
    Type resultType = IREE::Stream::ResourceType::get(
        getContext(), IREE::Stream::Lifetime::Constant);
    auto newOp = rewriter.create<IREE::Stream::TensorConstantOp>(
        constantOp.getLoc(), resultType,
        constantOp.value().cast<ElementsAttr>(),
        TypeAttr::get(constantOp.getType()),
        /*result_encoding_dims=*/ValueRange{},
        /*affinity=*/nullptr);

    Type unknownType = IREE::Stream::ResourceType::get(getContext());
    auto constantSize = rewriter.createOrFold<IREE::Stream::ResourceSizeOp>(
        constantOp.getLoc(), rewriter.getIndexType(), newOp.result());
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncTransferOp>(
        constantOp, unknownType, newOp.result(), constantSize, constantSize,
        /*source_affinity=*/nullptr,
        /*result_affinity=*/nullptr);
    return success();
  }
};

}  // namespace

void populateStandardConstantToStreamPatterns(
    MLIRContext *context, ConversionTarget &conversionTarget,
    TypeConverter &typeConverter, OwningRewritePatternList &patterns) {
  conversionTarget.addDynamicallyLegalOp<mlir::ConstantOp>(
      [](mlir::ConstantOp op) { return !op.getType().isa<TensorType>(); });

  patterns.insert<ConvertTensorConstantOp>(typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir
