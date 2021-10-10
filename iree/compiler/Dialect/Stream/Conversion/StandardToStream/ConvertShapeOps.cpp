// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

// Shapes go away here.
struct ElideTieShapeOp : public OpConversionPattern<Shape::TieShapeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      Shape::TieShapeOp op, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    Shape::TieShapeOpAdaptor operands(newOperands);
    rewriter.replaceOp(op, {operands.operand()});
    return success();
  }
};

}  // namespace

void populateStandardShapeToStreamPatterns(MLIRContext *context,
                                           ConversionTarget &conversionTarget,
                                           TypeConverter &typeConverter,
                                           OwningRewritePatternList &patterns) {
  conversionTarget.addIllegalOp<Shape::TieShapeOp>();

  patterns.insert<ElideTieShapeOp>(typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir
