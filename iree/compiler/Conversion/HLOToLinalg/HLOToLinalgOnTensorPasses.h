// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- HLOToLinalgOnTensorsPasses.h - Passes to convert from HLO To Linalg ===//
//
// IREE specific passes used in the HLO to Linalg on tensors conversion as well
// as fusion.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_CONVERSION_HLOTOLINALGONTENSORS_PASSES_H_
#define IREE_COMPILER_CONVERSION_HLOTOLINALGONTENSORS_PASSES_H_

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

/// Creates a pass to fuse operations on tensors.
std::unique_ptr<Pass> createFusionOfTensorOpsPass();

/// Creates XLA-HLO to Linalg on tensors transformation pass.
std::unique_ptr<OperationPass<FuncOp>> createHLOToLinalgOnTensorsPass(
    bool useLinalgOnTensorsPath = false);

/// Populates the patterns that convert from MHLO to Linalg on tensors. Imports
/// patterns from XLA, as well as some IREE specific modifications.
void populateHLOToLinalgOnTensorsConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    OwningRewritePatternList &patterns);

/// Populates IREE specific patterns to convert HLO broadcasting ops to Linalg.
/// These are being maintained separately because they are a standalone unit
/// that is both intricate and possible to upstream, should there be alignment
/// to do so.
void populateHLOBroadcastingToLinalgPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    OwningRewritePatternList &patterns);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_HLOTOLINALGONTENSORS_PASSES_H_