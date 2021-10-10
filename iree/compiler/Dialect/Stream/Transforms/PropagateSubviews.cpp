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
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Patterns.h"
#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-propagate-subviews"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {
namespace {

//===----------------------------------------------------------------------===//
// Global handling
//===----------------------------------------------------------------------===//

struct ExpandedGlobal {
  IREE::Util::GlobalOp resourceOp;
  IREE::Util::GlobalOp resourceSizeOp;
  IREE::Util::GlobalOp subviewOffsetOp;
  IREE::Util::GlobalOp subviewLengthOp;
};
using ExpandedGlobalMap = DenseMap<StringRef, ExpandedGlobal>;

// Expands each !stream.resource global in |rootOp| to have a matching
// parent resource size and subview range. Does not behave optimally if there
// already exist offset globals as duplicates will get added and we'll need to
// rely on global fusion to get rid of them. Note that this only expands globals
// and does not yet update use sites - we just need the ops to reference.
static ExpandedGlobalMap expandResourceGlobals(Operation *rootOp) {
  ExpandedGlobalMap expandedGlobals;

  // Gather all of the resource globals in the root.
  for (auto &region : rootOp->getRegions()) {
    for (auto globalOp : region.getOps<IREE::Util::GlobalOp>()) {
      if (!globalOp.type().isa<IREE::Stream::ResourceType>()) continue;
      expandedGlobals[globalOp.getName()].resourceOp = globalOp;
    }
  }

  // Expand each global by adding the offset right next to it.
  SymbolTable symbolTable(rootOp);
  auto indexType = IndexType::get(rootOp->getContext());
  for (auto &it : expandedGlobals) {
    auto &global = it.second;
    OpBuilder builder(global.resourceOp);

    auto sizeName = (global.resourceOp.getName() + "_storage_size").str();
    auto sizeOp = builder.create<IREE::Util::GlobalOp>(
        global.resourceOp.getLoc(), sizeName,
        /*isMutable=*/true, indexType);
    sizeOp.setVisibility(global.resourceOp.getVisibility());
    symbolTable.insert(sizeOp);
    global.resourceSizeOp = sizeOp;

    auto offsetName = (global.resourceOp.getName() + "_offset").str();
    auto offsetOp = builder.create<IREE::Util::GlobalOp>(
        global.resourceOp.getLoc(), offsetName,
        /*isMutable=*/true, indexType);
    offsetOp.setVisibility(global.resourceOp.getVisibility());
    symbolTable.insert(offsetOp);
    global.subviewOffsetOp = offsetOp;

    auto lengthName = (global.resourceOp.getName() + "_length").str();
    auto lengthOp = builder.create<IREE::Util::GlobalOp>(
        global.resourceOp.getLoc(), lengthName,
        /*isMutable=*/true, indexType);
    lengthOp.setVisibility(global.resourceOp.getVisibility());
    symbolTable.insert(lengthOp);
    global.subviewLengthOp = lengthOp;
  }

  return expandedGlobals;
}

//===----------------------------------------------------------------------===//
// Structural IR rewriting patterns
//===----------------------------------------------------------------------===//

static bool isResourceType(Type type) {
  return type.isa<IREE::Stream::ResourceType>();
}

static bool usesResources(Operation *op) {
  return llvm::any_of(op->getOperandTypes(), isResourceType) ||
         llvm::any_of(op->getResultTypes(), isResourceType);
}

static SmallVector<Type> expandTypes(TypeRange types) {
  if (types.empty()) return {};
  auto indexType = IndexType::get(types.front().getContext());
  SmallVector<Type> newTypes;
  newTypes.reserve(types.size() * 2);
  for (auto type : types) {
    newTypes.push_back(type);
    if (isResourceType(type)) {
      newTypes.push_back(indexType);  // resource size
      newTypes.push_back(indexType);  // subview offset
      newTypes.push_back(indexType);  // subview length
    }
  }
  return newTypes;
}

struct Subview {
  Value resource;
  Value resourceSize;
  Value subviewOffset;
  Value subviewLength;
};
using SubviewMap = llvm::DenseMap<Value, Subview>;

static Subview consumeSubview(Location loc, Value value, SubviewMap &subviewMap,
                              Value zeroOffset, OpBuilder &builder) {
  // DO NOT SUBMIT follow ties
  auto mapIt = subviewMap.find(value);
  if (mapIt != subviewMap.end()) {
    return mapIt->second;
  }

  if (auto subviewOp = dyn_cast_or_null<IREE::Stream::ResourceSubviewOp>(
          value.getDefiningOp())) {
    Subview subview;
    subview.resource = subviewOp.source();
    subview.resourceSize = subviewOp.source_size();
    subview.subviewOffset = subviewOp.source_offset();
    subview.subviewLength = subviewOp.result_size();
    return subview;
  } else {
    Subview subview;
    subview.resource = value;
    subview.resourceSize =
        IREE::Util::SizeAwareTypeInterface::queryValueSize(loc, value, builder);
    subview.subviewOffset = zeroOffset;
    subview.subviewLength = subview.resourceSize;
    return subview;
  }
}

static SmallVector<Value> expandOperands(Location loc, ValueRange operands,
                                         SubviewMap &subviewMap,
                                         Value zeroOffset, OpBuilder &builder) {
  SmallVector<Value> result;
  result.reserve(operands.size() * 2);
  for (auto operand : operands) {
    if (isResourceType(operand.getType())) {
      auto subview =
          consumeSubview(loc, operand, subviewMap, zeroOffset, builder);
      result.push_back(subview.resource);
      result.push_back(subview.resourceSize);
      result.push_back(subview.subviewOffset);
      result.push_back(subview.subviewLength);
    } else {
      result.push_back(operand);
    }
  }
  return result;
}

static void expandSubviews(Operation *op, ExpandedGlobalMap &globalMap,
                           Value zeroOffset, SubviewMap &subviewMap);

// Finds the size of a block argument resource or materializes a size if needed.
// The returned SSA value will be valid at the insertion point (by way of clones
// or other trickery required to make it so).
static Value makeBlockArgResourceSize(Location loc, Value resourceValue,
                                      OpBuilder &builder) {
  // We can take any implicitly captured SSA values.
  if (auto sizeAwareOp = dyn_cast_or_null<IREE::Util::SizeAwareOpInterface>(
          resourceValue.getDefiningOp())) {
    auto sizeValue = sizeAwareOp.getResultSizeFromValue(resourceValue);
    if (sizeValue) return sizeValue;
  }

  // Try first to scan uses in the IR. Since we carry the shape in most ops we
  // are likely to find at least some SSA value we can inspect.
  for (auto &use : resourceValue.getUses()) {
    auto sizeAwareOp =
        dyn_cast<IREE::Util::SizeAwareOpInterface>(use.getOwner());
    if (!sizeAwareOp) continue;
    auto sizeValue = sizeAwareOp.getOperandSize(use.getOperandNumber());
    if (!sizeValue) continue;
    if (sizeValue.getParentRegion()->isProperAncestor(
            builder.getInsertionBlock()->getParent())) {
      // Size value found and implicitly captured; we can reuse (could be
      // a parent block argument, a constant, computed, etc).
      return sizeValue;
    } else if (auto blockArg = sizeValue.dyn_cast<BlockArgument>()) {
      if (blockArg.getParentBlock()->isEntryBlock()) {
        // Dynamic dimension passed in to the entry block; safe to use.
        return sizeValue;
      }
    } else if (auto constantOp =
                   dyn_cast<mlir::ConstantOp>(sizeValue.getDefiningOp())) {
      // Constant op - duplicate at the builder location so we don't have to
      // worry about SSA dominance issues. CSE will clean up the dupes later.
      return builder.clone(*constantOp.getOperation())->getResult(0);
    }
    // Uninspectable value.
  }

  // If we couldn't find anything we could use we'll insert the size query. The
  // hope is that more program analysis could take care of this for us.
  return builder.create<IREE::Stream::ResourceSizeOp>(loc, resourceValue);
}

static void expandRegion(Region &region, ExpandedGlobalMap &globalMap,
                         Value zeroOffset, SubviewMap subviewMap) {
  // Update all block arguments.
  auto indexType = IndexType::get(region.getContext());
  for (auto &block : region.getBlocks()) {
    if (!llvm::any_of(block.getArgumentTypes(), isResourceType)) continue;

    // Insert and build a list of expanded (resource, size, offset) tuples.
    SmallVector<Subview> expansions;
    for (int i = block.getNumArguments() - 1; i >= 0; --i) {
      auto arg = block.getArgument(i);
      if (!isResourceType(arg.getType())) continue;
      Subview subview;
      subview.resource = arg;
      subview.resourceSize = block.insertArgument(i + 1, indexType);
      subview.subviewOffset = block.insertArgument(i + 2, indexType);
      subview.subviewLength = block.insertArgument(i + 3, indexType);
      expansions.push_back(subview);
      subviewMap[arg] = subview;
    }

    // Insert subviews that we've sunk from callers.
    auto builder = OpBuilder::atBlockBegin(&block);
    for (auto &expansion : expansions) {
      auto subviewOp = builder.create<IREE::Stream::ResourceSubviewOp>(
          region.getLoc(), expansion.resource, expansion.resourceSize,
          expansion.subviewOffset, expansion.subviewLength);
      expansion.resource.replaceAllUsesExcept(subviewOp.result(), subviewOp);
    }
  }

  // Walk blocks forward in domination order so that we add dominating values to
  // the offset map. Note that DominanceInfo is just determined not to be
  // cool about things when there's only one block so we have to special case.
  if (region.hasOneBlock()) {
    for (auto &op :
         llvm::make_early_inc_range(region.front().getOperations())) {
      expandSubviews(&op, globalMap, zeroOffset, subviewMap);
    }
  } else {
    DominanceInfo domInfo(region.getParentOp());
    for (auto *blockInfo : llvm::breadth_first(domInfo.getRootNode(&region))) {
      auto *block = blockInfo->getBlock();
      for (auto &op : llvm::make_early_inc_range(block->getOperations())) {
        expandSubviews(&op, globalMap, zeroOffset, subviewMap);
      }
    }
  }
}

// Moves resource subviews from global stores to loads.
// Requires that the ExpandGlobalStoreOp pattern elides the await.
//
// Example:
//  %0 = util.global.load @foo : !stream.resource
//  ->
//  %0 = util.global.load @foo : !stream.resource
//  %s = util.global.load @foo_size : index
//  %o = util.global.load @foo_offset : index
//  %l = util.global.load @foo_length : index
//  %1 = stream.resource.subview %0[%o] :
//       !stream.resource<*>{%s} -> !stream.resource<*>{%l}
static void expandGlobalLoadOp(IREE::Util::GlobalLoadOp op,
                               ExpandedGlobalMap &globalMap, Value zeroOffset,
                               SubviewMap &subviewMap) {
  if (!usesResources(op)) return;
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  auto indexType = builder.getIndexType();
  auto &expandedGlobal = globalMap[op.global()];
  Subview subview;
  subview.resource = op.result();
  subview.resourceSize =
      builder
          .create<IREE::Util::GlobalLoadOp>(
              op.getLoc(), indexType, expandedGlobal.resourceSizeOp.getName())
          .result();
  subview.subviewOffset =
      builder
          .create<IREE::Util::GlobalLoadOp>(
              op.getLoc(), indexType, expandedGlobal.subviewOffsetOp.getName())
          .result();
  subview.subviewLength =
      builder
          .create<IREE::Util::GlobalLoadOp>(
              op.getLoc(), indexType, expandedGlobal.subviewLengthOp.getName())
          .result();
  subviewMap[op.result()] = subview;
  auto subviewOp = builder.create<IREE::Stream::ResourceSubviewOp>(
      op.getLoc(), subview.resource, subview.resourceSize,
      subview.subviewOffset, subview.subviewLength);
  op.result().replaceAllUsesExcept(subviewOp.result(), subviewOp);
}

// Moves resource subviews from global stores to loads.
// Requires that the ExpandGlobalLoadOp pattern inserts the await.
//
// Example:
//  %1 = stream.resource.subview %0[%o] :
//       !stream.resource<*>{%s} -> !stream.resource<*>{%l}
//  util.global.store %1, @foo : !stream.resource
//  ->
//  util.global.store %0, @foo : !stream.resource
//  util.global.store %s, @foo_size : index
//  util.global.store %o, @foo_offset : index
//  util.global.store %l, @foo_length : index
static void expandGlobalStoreOp(IREE::Util::GlobalStoreOp op,
                                ExpandedGlobalMap &globalMap, Value zeroOffset,
                                SubviewMap &subviewMap) {
  if (!usesResources(op)) return;
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  auto subview =
      consumeSubview(op.getLoc(), op.value(), subviewMap, zeroOffset, builder);
  auto &expandedGlobal = globalMap[op.global()];
  builder.create<IREE::Util::GlobalStoreOp>(
      op.getLoc(), subview.resource, expandedGlobal.resourceOp.getName());
  builder.create<IREE::Util::GlobalStoreOp>(
      op.getLoc(), subview.resourceSize,
      expandedGlobal.resourceSizeOp.getName());
  builder.create<IREE::Util::GlobalStoreOp>(
      op.getLoc(), subview.subviewOffset,
      expandedGlobal.subviewOffsetOp.getName());
  builder.create<IREE::Util::GlobalStoreOp>(
      op.getLoc(), subview.subviewLength,
      expandedGlobal.subviewLengthOp.getName());
  op.erase();
}

static void expandInitializerOp(IREE::Util::InitializerOp op,
                                ExpandedGlobalMap &globalMap, Value zeroOffset,
                                SubviewMap &subviewMap) {
  expandRegion(op.getRegion(), globalMap, zeroOffset, subviewMap);
}

// Inserts subviews on resource arguments.
// Requires that the ExpandCallOp/ExpandReturnOp patterns handle migrating the
// await.
//
// NOTE: this needs IPO to remove redundant subviews in cases where the call
// sites don't need a wait.
//
// Example:
//  func @foo(%0: !stream.resource)
//  ->
//  func @foo(%0: !stream.resource, %sz: index, %o: index, %l: index) {
//    %1 = stream.resource.subview %0[%o] : {%sz} -> {%l}
static void expandFuncOp(mlir::FuncOp op, ExpandedGlobalMap &globalMap,
                         Value zeroOffset, SubviewMap &subviewMap) {
  auto oldType = op.getType();
  auto inputTypes = expandTypes(oldType.getInputs());
  auto resultTypes = expandTypes(oldType.getResults());
  auto newType = FunctionType::get(op.getContext(), inputTypes, resultTypes);
  if (newType != oldType) {
    op.setType(newType);
  }
  expandRegion(op.getRegion(), globalMap, zeroOffset, subviewMap);
}

// Splits resource operands and results into (resource, resourceSize,
// subviewOffset, subviewLength).
// Requires that the ExpandFuncOp/ExpandReturnOp patterns handle migrating the
// await.
//
// NOTE: this needs IPO to remove redundant values in cases where the call sites
// don't need a subview.
//
// Example:
//  %1 = stream.resource.subview %0[%o] : {%sz} -> {%l}
//  %r = call @foo(%1)
//  ->
//  %r, %rsz, %ro, %rl = call @foo(%0, %sz, %o, %l)
//  %2 = stream.resource.subview %r[%ro] : {%rsz} -> {%rl}
static void expandCallOp(mlir::CallOp op, Value zeroOffset,
                         SubviewMap &subviewMap) {
  if (!usesResources(op)) return;

  // Build the new call op with expanded operands and results.
  OpBuilder builder(op);
  auto operands = expandOperands(op.getLoc(), op.operands(), subviewMap,
                                 zeroOffset, builder);
  auto resultTypes = expandTypes(op.getResultTypes());
  auto newOp = builder.create<mlir::CallOp>(op.getLoc(), op.callee(),
                                            resultTypes, operands);

  // Insert subviews on results that we are sinking across the call edge.
  // The hope is that by moving the subviews here we can fold with uses inside
  // of this function.
  builder.setInsertionPointAfter(newOp);
  unsigned newIdx = 0;
  for (unsigned oldIdx = 0; oldIdx < op.getNumResults(); ++oldIdx) {
    auto oldResult = op.getResult(oldIdx);
    if (!isResourceType(oldResult.getType())) {
      auto newResult = newOp.getResult(newIdx++);
      oldResult.replaceAllUsesWith(newResult);
      continue;
    }
    Subview subview;
    subview.resource = newOp.getResult(newIdx++);
    subview.resourceSize = newOp.getResult(newIdx++);
    subview.subviewOffset = newOp.getResult(newIdx++);
    subview.subviewLength = newOp.getResult(newIdx++);
    subviewMap[subview.resource] = subview;
    auto subviewOp = builder.create<IREE::Stream::ResourceSubviewOp>(
        op.getLoc(), subview.resource, subview.resourceSize,
        subview.subviewOffset, subview.subviewLength);
    oldResult.replaceAllUsesWith(subviewOp.result());
  }

  op.erase();
}

// Moves subviews to callers upon return.
// Requires that the ExpandFuncOp/ExpandCallOp patterns handle migrating the
// await.
//
// Example:
//  %1 = stream.resource.subview %0[%o] : {%sz} -> {%l}
//  return %1
//  ->
//  return %0, %sz, %o, %l
static void expandReturnOp(mlir::ReturnOp op, Value zeroOffset,
                           SubviewMap &subviewMap) {
  if (!usesResources(op)) return;
  OpBuilder builder(op);
  auto operands = expandOperands(op.getLoc(), op.operands(), subviewMap,
                                 zeroOffset, builder);
  builder.create<mlir::ReturnOp>(op.getLoc(), operands);
  op.erase();
}

// Moves subviews across branches.
// Requires that the ExpandFuncOp pattern handles modifying the block args.
//
// Example:
//    %1 = stream.resource.subview %0[%o] : {%sz} -> {%l}
//    br ^bb1(%1)
//  ^bb1(%b):
//  ->
//    br ^bb1(%0, %sz, %o, %l)
//  ^bb1(%a, %b, %c, %d):
//    %1 = stream.resource.subview %a[%b] : {%c} -> {%d}
static void expandBranchOp(mlir::BranchOp op, Value zeroOffset,
                           SubviewMap &subviewMap) {
  OpBuilder builder(op);
  auto operands = expandOperands(op.getLoc(), op.destOperands(), subviewMap,
                                 zeroOffset, builder);
  builder.create<mlir::BranchOp>(op.getLoc(), op.dest(), operands);
  op.erase();
}
static void expandCondBranchOp(mlir::CondBranchOp op, Value zeroOffset,
                               SubviewMap &subviewMap) {
  if (!usesResources(op)) return;
  OpBuilder builder(op);
  builder.create<mlir::CondBranchOp>(
      op.getLoc(), op.condition(), op.trueDest(),
      expandOperands(op.getLoc(), op.trueDestOperands(), subviewMap, zeroOffset,
                     builder),
      op.falseDest(),
      expandOperands(op.getLoc(), op.falseDestOperands(), subviewMap,
                     zeroOffset, builder));
  op.erase();
}

static void expandSubviews(Operation *op, ExpandedGlobalMap &globalMap,
                           Value zeroOffset, SubviewMap &subviewMap) {
  if (auto loadOp = dyn_cast<IREE::Util::GlobalLoadOp>(op)) {
    expandGlobalLoadOp(loadOp, globalMap, zeroOffset, subviewMap);
  } else if (auto storeOp = dyn_cast<IREE::Util::GlobalStoreOp>(op)) {
    expandGlobalStoreOp(storeOp, globalMap, zeroOffset, subviewMap);
  } else if (auto initializerOp = dyn_cast<IREE::Util::InitializerOp>(op)) {
    expandInitializerOp(initializerOp, globalMap, zeroOffset, subviewMap);
  } else if (auto funcOp = dyn_cast<mlir::FuncOp>(op)) {
    expandFuncOp(funcOp, globalMap, zeroOffset, subviewMap);
  } else if (auto callOp = dyn_cast<mlir::CallOp>(op)) {
    expandCallOp(callOp, zeroOffset, subviewMap);
  } else if (auto returnOp = dyn_cast<mlir::ReturnOp>(op)) {
    expandReturnOp(returnOp, zeroOffset, subviewMap);
  } else if (auto branchOp = dyn_cast<mlir::BranchOp>(op)) {
    expandBranchOp(branchOp, zeroOffset, subviewMap);
  } else if (auto condBranchOp = dyn_cast<mlir::CondBranchOp>(op)) {
    expandCondBranchOp(condBranchOp, zeroOffset, subviewMap);
  }
}

//===----------------------------------------------------------------------===//
// -iree-stream-propagate-timepoints
//===----------------------------------------------------------------------===//

class PropagateSubviewsPass
    : public PropagateSubviewsBase<PropagateSubviewsPass> {
 public:
  PropagateSubviewsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto rootOp = getOperation();

    // Expand all util.global ops holding resources into resource and subview.
    auto globalMap = expandResourceGlobals(rootOp);

    // Walk the entire IR tree and expand the globals.
    // We could do this via pattern application but that gets much trickier to
    // manage with the expansion as we'd need to prevent ourselves from
    // expanding multiple times.
    for (auto callableOp : rootOp.getOps<CallableOpInterface>()) {
      auto zeroOffset =
          OpBuilder::atBlockBegin(&callableOp.getCallableRegion()->front())
              .create<ConstantIndexOp>(callableOp.getLoc(), 0);
      SubviewMap subviewMap;
      expandSubviews(callableOp, globalMap, zeroOffset, subviewMap);
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createPropagateSubviewsPass() {
  return std::make_unique<PropagateSubviewsPass>();
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
