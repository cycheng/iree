// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion2/StreamToHAL/ConvertStreamToHAL.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/Utils/DeviceSwitchBuilder.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

static Value makeElementType(Location loc, Type elementType,
                             OpBuilder &builder) {
  auto i32Value = IREE::HAL::getElementTypeValue(elementType);
  assert(i32Value.hasValue() && "unhandled element type for allocation");
  auto constantValue =
      builder.createOrFold<ConstantIntOp>(loc, i32Value.getValue(), 32);
  return constantValue;
}

static Value makeEncodingType(Location loc, Attribute encodingType,
                              OpBuilder &builder) {
  auto i32Value = IREE::HAL::getEncodingTypeValue(encodingType);
  assert(i32Value.hasValue() && "unhandled encoding type for allocation");
  auto constantValue =
      builder.createOrFold<ConstantIntOp>(loc, i32Value.getValue(), 32);
  return constantValue;
}

static Value lookupExecutableLayout(Location loc, Value device,
                                    IREE::HAL::InterfaceOp interfaceOp,
                                    OpBuilder &builder) {
  auto lookupOp = builder.create<IREE::HAL::ExecutableLayoutLookupOp>(
      loc, IREE::HAL::ExecutableLayoutType::get(loc.getContext()), device,
      interfaceOp.push_constantsAttr(),
      interfaceOp.getExecutableSetLayoutsAttr());
  return lookupOp.result();
}

static Value lookupDeviceFor(Operation *op, OpBuilder &builder) {
  // TODO(benvanik): make this do multi-device lookup and other fancy things.
  auto lookupOp = builder.create<IREE::HAL::ExSharedDeviceOp>(op->getLoc());
  return lookupOp.result();
}

static Value lookupAllocatorFor(Operation *op, OpBuilder &builder) {
  auto device = lookupDeviceFor(op, builder);
  auto allocatorOp =
      builder.create<IREE::HAL::DeviceAllocatorOp>(op->getLoc(), device);
  return allocatorOp.result();
}

// Scans all of the stream.cmd.* ops in the region to derive a command category.
static IREE::HAL::CommandCategoryBitfield deriveCommandCategories(
    Region &region) {
  auto bits = IREE::HAL::CommandCategoryBitfield::None;
  for (auto &block : region) {
    for (auto &op : block) {
      if (isa<IREE::Stream::CmdDispatchOp>(op)) {
        bits = bits | IREE::HAL::CommandCategoryBitfield::Dispatch;
      } else {
        bits = bits | IREE::HAL::CommandCategoryBitfield::Transfer;
      }
      for (auto &nestedRegion : op.getRegions()) {
        bits = bits | deriveCommandCategories(nestedRegion);
      }
    }
  }
  return bits;
}

class StreamConversionMapping {
 public:
  // Maps the stream dialect |executeOp| to the hal dialect |commandBuffer|
  // value used during recording. Patterns can use this to find the SSA value
  // they need to make hal.command_buffer.* ops.
  void mapCommandBuffer(IREE::Stream::CmdExecuteOp executeOp,
                        Value commandBuffer) {
    auto it = commandBuffers.insert(std::make_pair(executeOp, commandBuffer));
    assert(it.second &&
           "multiple command buffers cannot be registered for the same op");

    // Map all ops nested within the command buffer so we can query later.
    executeOp.walk([&](Operation *op) {
      commandBuffers.insert(std::make_pair(op, commandBuffer));
      return WalkResult::advance();
    });
  }

  // Looks up a mapped command buffer SSA value that can be used by the given
  // stream.cmd.* op.
  Value lookupCommandBufferFor(Operation *cmdOp) const {
    auto it = commandBuffers.find(cmdOp);
    assert(it != commandBuffers.end() &&
           "command buffer must have been registered during conversion");
    return it->second;
  }

 private:
  // Ops within stream.cmd.execute ops -> !hal.command_buffer.
  DenseMap<Operation *, Value> commandBuffers;
};

template <typename OpT>
class StreamConversionPattern : public OpConversionPattern<OpT> {
 public:
  StreamConversionPattern(std::shared_ptr<StreamConversionMapping> mapping,
                          TypeConverter &typeConverter, MLIRContext *context,
                          PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit),
        mapping(std::move(mapping)) {}

  virtual LogicalResult matchAndRewrite(
      OpT op, typename OpT::Adaptor operands,
      ConversionPatternRewriter &rewriter) const = 0;

 protected:
  LogicalResult matchAndRewrite(
      OpT op, llvm::ArrayRef<Value> rawOperands,
      ConversionPatternRewriter &rewriter) const override {
    OpT::Adaptor operands(rawOperands, op->getAttrDictionary(),
                          op->getRegions());
    return matchAndRewrite(op, operands, rewriter);
  }

  std::shared_ptr<StreamConversionMapping> mapping;
};

struct ResourceAllocOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceAllocOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceAllocOp allocOp,
      IREE::Stream::ResourceAllocOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    auto allocator = lookupAllocatorFor(allocOp, rewriter);
    auto bufferType = rewriter.getType<IREE::HAL::BufferType>();

    SmallVector<Value> results;
    for (auto it : llvm::zip(allocOp.results(), allocOp.storage_sizes())) {
      auto resourceResult = std::get<0>(it);
      auto resourceType =
          resourceResult.getType().cast<IREE::Stream::ResourceType>();
      auto storageSize = std::get<1>(it);

      auto memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
      auto bufferUsage = IREE::HAL::BufferUsageBitfield::None;
      switch (resourceType.getLifetime()) {
        default:
          return allocOp.emitOpError()
                 << "unsupported resource lifetime: "
                 << IREE::Stream::stringifyLifetime(resourceType.getLifetime());
        case IREE::Stream::Lifetime::Constant:
          // Device local; copies required to get into external resources.
          memoryTypes =
              memoryTypes | IREE::HAL::MemoryTypeBitfield::DeviceLocal;
          bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Constant;
          break;
        case IREE::Stream::Lifetime::Variable:
          // Device local; copies required to get into external resources.
          memoryTypes =
              memoryTypes | IREE::HAL::MemoryTypeBitfield::DeviceLocal;
          break;
        case IREE::Stream::Lifetime::External:
          // #yolo; these come from/go to outside the program.
          // Today we assume they are device-local|host-visible just for
          // practical purposes but that does not have to be true. We really
          // want this to be something we analyze and handle on the edges
          // (transfering devices/etc if needed).
          memoryTypes = memoryTypes |
                        IREE::HAL::MemoryTypeBitfield::DeviceLocal |
                        IREE::HAL::MemoryTypeBitfield::HostVisible;
          // NOTE: we may not map it but users may after they get them back.
          // Another reason we should annotate this - having a buffer be
          // mappable is potentially expensive (may get a 2nd copy in memory!).
          bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Mapping;
          break;
        case IREE::Stream::Lifetime::Staging:
          // Host local; copies required to get into device resources.
          // We could vary this based on staging usage (upload/download) by
          // making it device-local|host-visible, but host-local means we have
          // a better chance of mapping it during uploads.
          memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::HostLocal |
                        IREE::HAL::MemoryTypeBitfield::DeviceVisible;
          bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Transfer |
                        IREE::HAL::BufferUsageBitfield::Mapping;
          break;
        case IREE::Stream::Lifetime::Transient:
          // Device local; copies required to get into external resources.
          memoryTypes = memoryTypes |
                        IREE::HAL::MemoryTypeBitfield::DeviceLocal |
                        IREE::HAL::MemoryTypeBitfield::Transient;
          break;
      }

      // TODO(benvanik): refine usage based on analysis.
      bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Transfer |
                    IREE::HAL::BufferUsageBitfield::Dispatch;

      auto allocateOp = rewriter.create<IREE::HAL::AllocatorAllocateOp>(
          allocOp.getLoc(), bufferType, allocator, memoryTypes, bufferUsage,
          storageSize);
      results.push_back(allocateOp.result());
    }

    rewriter.replaceOp(allocOp, results);
    return success();
  }
};

struct ResourceAllocaOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceAllocaOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceAllocaOp allocaOp,
      IREE::Stream::ResourceAllocaOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    auto allocator = lookupAllocatorFor(allocaOp, rewriter);
    auto resourceType =
        allocaOp.result().getType().cast<IREE::Stream::ResourceType>();
    auto bufferType = rewriter.getType<IREE::HAL::BufferType>();

    // Transient allocations are device-local. Copies are required to get their
    // contents back on the host/another device.
    auto memoryTypes = IREE::HAL::MemoryTypeBitfield::DeviceLocal |
                       IREE::HAL::MemoryTypeBitfield::Transient;

    // TODO(benvanik): refine usage.
    // We know by construction that transient buffers are not host visible and
    // as such can only be used for device commands. We should be able to more
    // closely limit to just dispatch or transfer though.
    auto bufferUsage = IREE::HAL::BufferUsageBitfield::Dispatch |
                       IREE::HAL::BufferUsageBitfield::Transfer;

    auto allocateOp = rewriter.create<IREE::HAL::AllocatorAllocateOp>(
        allocaOp.getLoc(), bufferType, allocator, memoryTypes, bufferUsage,
        allocaOp.storage_size());

    // TODO(benvanik): stream ordered allocations.
    auto resolvedTimepoint =
        rewriter.create<mlir::ConstantIndexOp>(allocaOp.getLoc(), 0)
            .getResult();

    rewriter.replaceOp(allocaOp, {allocateOp.result(), resolvedTimepoint});
    return success();
  }
};

struct ResourceDeallocaOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceDeallocaOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceDeallocaOp deallocaOp,
      IREE::Stream::ResourceDeallocaOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): stream ordered allocations.
    rewriter.eraseOp(deallocaOp);
    return success();
  }
};

struct ResourceSizeOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceSizeOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceSizeOp sizeOp,
      IREE::Stream::ResourceSizeOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferLengthOp>(
        sizeOp, rewriter.getIndexType(), operands.operand());
    return success();
  }
};

struct ResourceMapOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceMapOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceMapOp mapOp,
      IREE::Stream::ResourceMapOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    auto allocator = lookupAllocatorFor(mapOp, rewriter);
    auto bufferType = rewriter.getType<IREE::HAL::BufferType>();

    // We know this is a staging buffer. We could refine usage here by seeing
    // whether this was upload or download.
    auto memoryTypes = IREE::HAL::MemoryTypeBitfield::HostLocal |
                       IREE::HAL::MemoryTypeBitfield::DeviceVisible;
    auto bufferUsage = IREE::HAL::BufferUsageBitfield::Mapping |
                       IREE::HAL::BufferUsageBitfield::Transfer;

    rewriter.replaceOpWithNewOp<IREE::HAL::AllocatorMapOp>(
        mapOp, bufferType, allocator, memoryTypes, bufferUsage,
        operands.source(), operands.source_offset(), operands.result_size());
    return success();
  }
};

struct ResourceTryMapOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceTryMapOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceTryMapOp tryMapOp,
      IREE::Stream::ResourceTryMapOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    auto allocator = lookupAllocatorFor(tryMapOp, rewriter);
    auto resourceType =
        tryMapOp.result().getType().cast<IREE::Stream::ResourceType>();
    auto bufferType = rewriter.getType<IREE::HAL::BufferType>();

    auto memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
    auto bufferUsage = IREE::HAL::BufferUsageBitfield::None;
    switch (resourceType.getLifetime()) {
      default:
        return tryMapOp.emitOpError()
               << "unsupported resource lifetime: "
               << IREE::Stream::stringifyLifetime(resourceType.getLifetime());
      case IREE::Stream::Lifetime::Constant:
        // Device local; copies required to get into external resources.
        memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::DeviceLocal;
        bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Constant;
        break;
      case IREE::Stream::Lifetime::Staging:
        // Host local; copies required to get into device resources.
        // We could vary this based on staging usage (upload/download) by
        // making it device-local|host-visible, but host-local means we have
        // a better chance of mapping it during uploads.
        memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::HostLocal |
                      IREE::HAL::MemoryTypeBitfield::DeviceVisible;
        bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Transfer |
                      IREE::HAL::BufferUsageBitfield::Mapping;
        break;
    }

    // TODO(benvanik): refine usage based on analysis.
    bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Transfer |
                  IREE::HAL::BufferUsageBitfield::Dispatch;

    rewriter.replaceOpWithNewOp<IREE::HAL::AllocatorTryMapOp>(
        tryMapOp, rewriter.getI1Type(), bufferType, allocator, memoryTypes,
        bufferUsage, operands.source(), operands.source_offset(),
        operands.result_size());
    return success();
  }
};

struct ResourceLoadOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceLoadOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceLoadOp loadOp,
      IREE::Stream::ResourceLoadOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loadType = getTypeConverter()->convertType(loadOp.result().getType());
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferLoadOp>(
        loadOp, loadType, operands.source(), operands.source_offset());
    return success();
  }
};

struct ResourceStoreOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceStoreOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceStoreOp storeOp,
      IREE::Stream::ResourceStoreOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.create<IREE::HAL::BufferStoreOp>(
        storeOp.getLoc(), operands.value(), operands.target(),
        operands.target_offset());
    rewriter.replaceOp(storeOp, operands.target());
    return success();
  }
};

struct ResourceSubviewOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceSubviewOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceSubviewOp subviewOp,
      IREE::Stream::ResourceSubviewOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    auto bufferType = rewriter.getType<IREE::HAL::BufferType>();
    // NOTE: this aliases! We assume at this point all useful alias analysis
    // has been performed and it's fine to lose the tie information here.
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferSubspanOp>(
        subviewOp, bufferType, operands.source(), operands.source_offset(),
        operands.result_size());
    return success();
  }
};

struct TensorImportOpPattern
    : public StreamConversionPattern<IREE::Stream::TensorImportOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::TensorImportOp importOp,
      IREE::Stream::TensorImportOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = importOp.getLoc();

    auto bufferView = operands.source();
    auto bufferType = rewriter.getType<IREE::HAL::BufferType>();
    auto bufferOp = rewriter.replaceOpWithNewOp<IREE::HAL::BufferViewBufferOp>(
        importOp, bufferType, bufferView);

    // Assert the shape of the buffer view matches the expected encoding shape.
    // NOTE: we do this before the other checks as it's the most likely mistake
    // and it's better to know of a shape mismatch than just buffer byte length
    // difference.
    buildEncodingAssertions(
        loc, bufferView,
        operands.result_encoding().getValue().cast<RankedTensorType>(),
        operands.result_encoding_dims(), rewriter);

    // Ensure we have enough bytes in the buffer for the encoding we have.
    // Note that having more bytes is fine:
    //   assert(expected_length <= actual_length);
    auto expectedLength = operands.result_size();
    auto actualLength = rewriter.create<IREE::HAL::BufferLengthOp>(
        loc, rewriter.getIndexType(), bufferOp);
    auto isMinSize = rewriter.create<mlir::CmpIOp>(
        loc, CmpIPredicate::ule, expectedLength, actualLength);
    rewriter.create<mlir::AssertOp>(
        loc, isMinSize,
        "imported tensor buffer truncated; must be >= the encoded size");

    // TODO(benvanik): assert that the buffer view is accessible from the
    // target device. We need to add a hal.allocator.* op set for importing
    // or verifying that things are possible. To start we could just compare if
    // the allocators are identical - though we don't have the ops to do that.
    auto sourceAllocatorOp = rewriter.create<IREE::HAL::BufferAllocatorOp>(
        loc, rewriter.getType<IREE::HAL::AllocatorType>(), bufferOp);
    auto targetAllocatorOp = lookupAllocatorFor(importOp, rewriter);
    auto isAllocatorEq = rewriter.create<IREE::Util::CmpEQOp>(
        loc, rewriter.getI1Type(), sourceAllocatorOp, targetAllocatorOp);
    rewriter.create<mlir::AssertOp>(
        loc, isAllocatorEq,
        "imported tensor buffer allocator mismatch; must be from the same "
        "allocator (today)");

    return success();
  }

  // Inserts IR to assert that the buffer view shape and encoding matches the
  // expected encoding we have in the program. This ensures that the user didn't
  // pass a 4x8xf32 when they originally compiled the model for a 2x8x1xi8.
  //
  // Order: encoding (dense/etc), element type (f32/etc), rank, dims
  static void buildEncodingAssertions(Location loc, Value bufferView,
                                      RankedTensorType tensorType,
                                      ValueRange dynamicDims,
                                      OpBuilder &builder) {
    auto indexType = builder.getIndexType();
    auto i32Type = builder.getI32Type();

    // Encoding:
    {
      // NOTE: we should have verified supported encodings at entry into the HAL
      // pipeline.
      auto encodingType =
          IREE::HAL::getEncodingTypeValue(tensorType.getEncoding());
      assert(encodingType.hasValue() && "invalid tensor encoding");

      auto actualEncodingType =
          builder
              .create<IREE::HAL::BufferViewEncodingTypeOp>(loc, i32Type,
                                                           bufferView)
              .result();
      auto expectedEncodingType =
          builder.create<mlir::ConstantIntOp>(loc, encodingType.getValue(), 32);
      auto cmpOp = builder.create<mlir::CmpIOp>(
          loc, CmpIPredicate::eq, actualEncodingType, expectedEncodingType);
      std::string message;
      llvm::raw_string_ostream os(message);
      os << "encoding mismatch; expected ";
      if (tensorType.getEncoding()) {
        os << tensorType.getEncoding();
      } else {
        os << "dense";
      }
      os << " (for " << tensorType << ")";
      builder.create<mlir::AssertOp>(loc, cmpOp.result(), os.str());
    }

    // Element type:
    {
      // NOTE: we should have verified supported element types at entry into the
      // HAL pipeline.
      auto elementType =
          IREE::HAL::getElementTypeValue(tensorType.getElementType());
      assert(elementType.hasValue() && "invalid tensor element type");

      auto actualElementType = builder
                                   .create<IREE::HAL::BufferViewElementTypeOp>(
                                       loc, i32Type, bufferView)
                                   .result();
      auto expectedElementType =
          builder.create<mlir::ConstantIntOp>(loc, elementType.getValue(), 32);
      auto cmpOp = builder.create<mlir::CmpIOp>(
          loc, CmpIPredicate::eq, actualElementType, expectedElementType);
      std::string message;
      llvm::raw_string_ostream os(message);
      os << "element type mismatch; expected " << tensorType.getElementType();
      builder.create<mlir::AssertOp>(loc, cmpOp.result(), os.str());
    }

    // Rank:
    {
      auto actualRank =
          builder
              .create<IREE::HAL::BufferViewRankOp>(loc, indexType, bufferView)
              .result();
      auto expectedRank =
          builder.create<mlir::ConstantIndexOp>(loc, tensorType.getRank());
      auto cmpOp = builder.create<mlir::CmpIOp>(loc, CmpIPredicate::eq,
                                                actualRank, expectedRank);
      std::string message;
      llvm::raw_string_ostream os(message);
      os << "rank mismatch; expected rank " << tensorType.getRank() << " (for "
         << tensorType << ")";
      builder.create<mlir::AssertOp>(loc, cmpOp.result(), os.str());
    }

    // Compare each dim in turn.
    {
      unsigned dynamicIdx = 0;
      Value shapeOk;
      for (int64_t idx = 0; idx < tensorType.getRank(); ++idx) {
        auto actualDim = builder.create<IREE::HAL::BufferViewDimOp>(
            loc, indexType, bufferView, builder.getIndexAttr(idx));
        Value expectedDim;
        if (tensorType.isDynamicDim(idx)) {
          expectedDim = dynamicDims[dynamicIdx++];
        } else {
          expectedDim = builder.create<mlir::ConstantIndexOp>(
              loc, tensorType.getDimSize(idx));
        }
        auto cmpOp = builder.create<mlir::CmpIOp>(loc, CmpIPredicate::eq,
                                                  actualDim, expectedDim);
        if (shapeOk) {
          shapeOk = builder.create<mlir::OrOp>(loc, shapeOk, cmpOp);
        } else {
          shapeOk = cmpOp;
        }
      }
      std::string message;
      llvm::raw_string_ostream os(message);
      os << "shape mismatch (for " << tensorType << ")";
      builder.create<mlir::AssertOp>(loc, shapeOk, os.str());
    }
  }
};

struct TensorExportOpPattern
    : public StreamConversionPattern<IREE::Stream::TensorExportOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::TensorExportOp exportOp,
      IREE::Stream::TensorExportOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = exportOp.getLoc();
    auto indexType = rewriter.getIndexType();
    auto i32Type = rewriter.getI32Type();
    auto tensorType =
        operands.source_encoding().getValue().cast<RankedTensorType>();
    auto dynamicDims = operands.source_encoding_dims();

    // NOTE: we should have verified supported encodings/types at entry into the
    // HAL pipeline.
    auto encodingType =
        IREE::HAL::getEncodingTypeValue(tensorType.getEncoding());
    assert(encodingType.hasValue() && "invalid tensor encoding");
    auto elementType =
        IREE::HAL::getElementTypeValue(tensorType.getElementType());
    assert(elementType.hasValue() && "invalid tensor element type");

    // Flatten static + dynamic shape dimensions.
    SmallVector<Value> dims;
    unsigned dynamicIdx = 0;
    for (int64_t idx = 0; idx < tensorType.getRank(); ++idx) {
      if (tensorType.isDynamicDim(idx)) {
        dims.push_back(dynamicDims[dynamicIdx++]);
      } else {
        dims.push_back(rewriter.create<mlir::ConstantIndexOp>(
            loc, tensorType.getDimSize(idx)));
      }
    }

    rewriter.replaceOpWithNewOp<IREE::HAL::BufferViewCreateOp>(
        exportOp, operands.source(), elementType.getValue(),
        encodingType.getValue(), dims);
    return success();
  }
};

struct CmdFlushOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdFlushOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdFlushOp op, IREE::Stream::CmdFlushOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): HAL command buffer op for flush.
    rewriter.eraseOp(op);
    return success();
  }
};

struct CmdInvalidateOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdInvalidateOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdInvalidateOp op,
      IREE::Stream::CmdInvalidateOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): HAL command buffer op for invalidate.
    rewriter.eraseOp(op);
    return success();
  }
};

struct CmdDiscardOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdDiscardOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdDiscardOp op,
      IREE::Stream::CmdDiscardOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): HAL command buffer op for discard.
    rewriter.eraseOp(op);
    return success();
  }
};

struct CmdFillOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdFillOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdFillOp fillOp, IREE::Stream::CmdFillOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    auto pattern =
        splatFillPattern(fillOp.getLoc(), operands.value(), rewriter);
    if (!pattern) {
      return fillOp.emitError() << ">4 byte/non-byte-aligned fills are not yet "
                                   "implemented (require special emulation)";
    }
    auto commandBuffer = mapping->lookupCommandBufferFor(fillOp);
    rewriter.replaceOpWithNewOp<IREE::HAL::CommandBufferFillBufferOp>(
        fillOp, commandBuffer, operands.target(), operands.target_offset(),
        operands.target_length(), pattern);
    return success();
  }

  // Splats a pattern value of 1, 2, or 4 bytes out to a 4 byte integer value.
  // The bit representation of |baseValue| will be repeated as many times as
  // needed in the returned value to use 4 bytes of storage. For example,
  // a 16-bit value (int or float) will have its native bit representation
  // repeated twice.
  static Value splatFillPattern(Location loc, Value baseValue,
                                OpBuilder &builder) {
    // Bitcast to an integer, then use integer math for the rest of the pattern.
    auto baseBitWidth = baseValue.getType().getIntOrFloatBitWidth();
    baseValue = builder.createOrFold<BitcastOp>(
        loc, builder.getIntegerType(baseBitWidth), baseValue);
    switch (baseBitWidth) {
      case 8: {
        // (v << 24) | (v << 16) | (v << 8) | v
        auto b0 = builder.createOrFold<ZeroExtendIOp>(
            loc, baseValue, builder.getIntegerType(32));
        auto c8 = builder.create<ConstantIntOp>(loc, 8, 32);
        auto b1 = builder.createOrFold<ShiftLeftOp>(loc, b0, c8);
        auto c16 = builder.create<ConstantIntOp>(loc, 16, 32);
        auto b2 = builder.createOrFold<ShiftLeftOp>(loc, b0, c16);
        auto c24 = builder.create<ConstantIntOp>(loc, 24, 32);
        auto b3 = builder.createOrFold<ShiftLeftOp>(loc, b0, c24);
        return builder.createOrFold<OrOp>(
            loc, b0,
            builder.createOrFold<OrOp>(
                loc, b1, builder.createOrFold<OrOp>(loc, b2, b3)));
      }
      case 16: {
        // (v << 16) | v
        auto c16 = builder.create<ConstantIntOp>(loc, 16, 32);
        auto b0 = builder.createOrFold<ZeroExtendIOp>(
            loc, baseValue, builder.getIntegerType(32));
        auto b1 = builder.createOrFold<ShiftLeftOp>(loc, b0, c16);
        return builder.createOrFold<OrOp>(loc, b0, b1);
      }
      case 32:
        return baseValue;
      default:
        return {};  // Unsupported (so far)
    }
  }
};

struct CmdCopyOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdCopyOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdCopyOp op, IREE::Stream::CmdCopyOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    auto commandBuffer = mapping->lookupCommandBufferFor(op);
    rewriter.replaceOpWithNewOp<IREE::HAL::CommandBufferCopyBufferOp>(
        op, commandBuffer, operands.source(), operands.source_offset(),
        operands.target(), operands.target_offset(), operands.length());
    return success();
  }
};

struct CmdDispatchOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdDispatchOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdDispatchOp dispatchOp,
      IREE::Stream::CmdDispatchOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = dispatchOp.getLoc();
    auto commandBuffer = mapping->lookupCommandBufferFor(dispatchOp);

    // Get the device handle we're executing against in this execution region.
    // Note that this is a dynamic value: we have to treat the device as unknown
    // here.
    auto device = rewriter.create<IREE::HAL::CommandBufferDeviceOp>(
        loc, rewriter.getType<IREE::HAL::DeviceType>(), commandBuffer);

    // Get the handle to the executable that is compatible with our device.
    auto executableOp =
        cast<IREE::HAL::ExecutableOp>(SymbolTable::lookupNearestSymbolFrom(
            dispatchOp, dispatchOp.entry_point().getRootReference()));
    assert(executableOp && "dispatch target executable op not found");

    // Ask each target backend to record their dispatch logic.
    IREE::HAL::DeviceSwitchRewriter switchRewriter(loc,
                                                   /*resultTypes=*/TypeRange{},
                                                   device, rewriter);
    for (auto variantOp :
         executableOp.getOps<IREE::HAL::ExecutableVariantOp>()) {
      auto entryPointOps =
          variantOp.getOps<IREE::HAL::ExecutableEntryPointOp>();
      auto entryPointIt = llvm::find_if(
          entryPointOps, [&](IREE::HAL::ExecutableEntryPointOp op) {
            return op.getNameAttr() ==
                   dispatchOp.entry_point().getLeafReference();
          });
      if (entryPointIt == entryPointOps.end()) {
        return variantOp.emitError()
               << "hal.executable.variant is missing the flow entry point for "
               << dispatchOp.entry_point();
      }
      auto entryPointOp = *entryPointIt;
      auto interfaceOp =
          dyn_cast<IREE::HAL::InterfaceOp>(SymbolTable::lookupSymbolIn(
              executableOp, entryPointOp.interfaceAttr()));

      auto *region = switchRewriter.addConditionRegion(
          variantOp.target().getMatchExpression());
      auto &entryBlock = region->front();
      auto caseBuilder = OpBuilder::atBlockBegin(&entryBlock);

      // Record push constants and buffer bindings.
      recordParameters(loc, device, commandBuffer, dispatchOp, operands,
                       interfaceOp, caseBuilder);

      // Dispatch with a target-specific workgroup count.
      auto entryPointSymRef =
          SymbolRefAttr::get(caseBuilder.getContext(), executableOp.getName(),
                             {SymbolRefAttr::get(entryPointOp->getParentOp()),
                              SymbolRefAttr::get(entryPointOp)});
      auto caseWorkgroupCount = calculateDispatchWorkgroupCount(
          loc, executableOp, entryPointOp, operands.workgroup_count(),
          caseBuilder);
      caseBuilder.create<IREE::HAL::CommandBufferDispatchSymbolOp>(
          loc, commandBuffer, entryPointSymRef, caseWorkgroupCount[0],
          caseWorkgroupCount[1], caseWorkgroupCount[2]);

      caseBuilder.create<IREE::HAL::ReturnOp>(loc);
    }
    switchRewriter.build();

    rewriter.eraseOp(dispatchOp);
    return success();
  }

  void recordParameters(Location loc, Value device, Value commandBuffer,
                        IREE::Stream::CmdDispatchOp dispatchOp,
                        IREE::Stream::CmdDispatchOp::Adaptor operands,
                        IREE::HAL::InterfaceOp interfaceOp,
                        OpBuilder &builder) const {
    auto executableLayout =
        lookupExecutableLayout(loc, device, interfaceOp, builder);

    // Push constant values.
    // TODO(#5322): symbolic push constant names on the hal.interface so we can
    // sparsely pack these.
    if (!operands.operands().empty()) {
      int pushConstantBase = 0;  // always 0 today
      SmallVector<Value> pushConstants;
      for (auto operand : operands.operands()) {
        // Need an explicit index cast to i32 since the
        // CommandBufferPushConstantsOp is intrinsically i32 based.
        // TODO(benvanik): don't force conversion yet - or do so
        // target-dependently.
        if (operand.getType().isa<IndexType>()) {
          pushConstants.push_back(builder.create<mlir::IndexCastOp>(
              dispatchOp.getLoc(), builder.getIntegerType(32), operand));
        } else {
          assert(
              (operand.getType().isInteger(32) || operand.getType().isF32()) &&
              "expected a 32-bit value");
          pushConstants.push_back(operand);
        }
      }
      builder.create<IREE::HAL::CommandBufferPushConstantsOp>(
          loc, commandBuffer, executableLayout,
          builder.getIndexAttr(pushConstantBase), pushConstants);
    }

    // TODO(benvanik): typed accessors for bindings.
    auto bindingSymbols = dispatchOp->getAttr("hal.interface.bindings")
                              .dyn_cast_or_null<ArrayAttr>();
    assert(bindingSymbols &&
           "interface materialization must annotate dispatch sites");
    auto bindingOps = llvm::to_vector<
        4>(llvm::map_range(bindingSymbols, [&](Attribute symRefAttr) {
      auto bindingOp =
          SymbolTable::lookupNearestSymbolFrom<IREE::HAL::InterfaceBindingOp>(
              dispatchOp, symRefAttr.cast<SymbolRefAttr>());
      assert(bindingOp && "binding not found");
      return bindingOp;
    }));
    // Sort in set -> binding order ascending.
    llvm::sort(bindingOps, [](IREE::HAL::InterfaceBindingOp lhs,
                              IREE::HAL::InterfaceBindingOp rhs) {
      int64_t lhsSet = lhs.set().getSExtValue();
      int64_t rhsSet = rhs.set().getSExtValue();
      if (lhsSet < rhsSet) return true;
      if (rhsSet > lhsSet) return false;
      int64_t lhsBinding = lhs.binding().getSExtValue();
      int64_t rhsBinding = rhs.binding().getSExtValue();
      return lhsBinding < rhsBinding;
    });

    // Push descriptor bindings.
    int64_t currentSet = -1;
    SmallVector<IREE::HAL::DescriptorSetBindingValue> bindings;
    auto flushSet = [&]() {
      builder.create<IREE::HAL::CommandBufferPushDescriptorSetOp>(
          loc, commandBuffer, executableLayout, currentSet, bindings);
      bindings.clear();
    };
    for (unsigned i = 0; i < operands.resources().size(); ++i) {
      auto bindingOp = bindingOps[i];
      int64_t set = bindingOp.set().getSExtValue();
      if (currentSet != -1 && currentSet != set) flushSet();
      currentSet = set;
      IREE::HAL::DescriptorSetBindingValue binding;
      binding.ordinal = builder.create<mlir::ConstantIndexOp>(
          loc, bindingOp.binding().getSExtValue());
      binding.buffer = operands.resources()[i];
      binding.byteOffset = operands.resource_offsets()[i];
      binding.byteLength = operands.resource_lengths()[i];
      bindings.push_back(binding);
    }
    if (currentSet != -1) flushSet();
  }

  // Calculates the workgroup count (x, y, z) for dispatching to the given
  // |entryPointOp|. The provided N-dimensional |workload| is the total number
  // of invocations required as calculated by the generic workload logic
  // (basically, number of output elements in tensors).
  static std::array<Value, 3> calculateDispatchWorkgroupCount(
      Location loc, IREE::HAL::ExecutableOp executableOp,
      IREE::HAL::ExecutableEntryPointOp entryPointOp, ValueRange workload,
      OpBuilder &builder) {
    Region *region = entryPointOp.getBody();
    if (region) {
      return calculateDispatchWorkgroupCountFromRegion(loc, entryPointOp,
                                                       workload, builder);
    }
    auto workgroupSize = calculateDispatchWorkgroupSize(
        loc, executableOp, entryPointOp, workload, builder);
    return calculateWorkloadWorkgroupCount(loc, workload, workgroupSize,
                                           builder);
  }

  // Calculates the workgroup size (x, y, z). These are the dimension numbers
  // for a single workgroup.
  static std::array<Value, 3> calculateDispatchWorkgroupSize(
      Location loc, IREE::HAL::ExecutableOp executableOp,
      IREE::HAL::ExecutableEntryPointOp entryPointOp, ValueRange workload,
      OpBuilder &builder) {
    // When no workgroup size is specified we just assume [1,1,1].
    // This yields a workgroup count that models the extents of the workload.
    return {
        builder.createOrFold<mlir::ConstantIndexOp>(loc, 1),
        builder.createOrFold<mlir::ConstantIndexOp>(loc, 1),
        builder.createOrFold<mlir::ConstantIndexOp>(loc, 1),
    };
  }

  static std::array<Value, 3> calculateDispatchWorkgroupCountFromRegion(
      Location loc, IREE::HAL::ExecutableEntryPointOp entryPointOp,
      ValueRange workload, OpBuilder &builder) {
    // TODO(benvanik): replace with region inlining util.
    Block *body = entryPointOp.getBlock();
    BlockAndValueMapping bvm;
    for (auto args : llvm::enumerate(workload)) {
      bvm.map(body->getArgument(args.index()), args.value());
    }
    for (Operation &op : body->without_terminator()) {
      builder.clone(op, bvm);
    }
    auto returnOp = cast<IREE::HAL::ReturnOp>(body->getTerminator());
    return {
        bvm.lookup(returnOp.operands()[0]),
        bvm.lookup(returnOp.operands()[1]),
        bvm.lookup(returnOp.operands()[2]),
    };
  }

  // Calculates the workgroup count (x, y, z) given the total N-dimensional
  // |workload| and specific |workgroupSize|.
  static std::array<Value, 3> calculateWorkloadWorkgroupCount(
      Location loc, ValueRange workload,
      const std::array<Value, 3> &workgroupSize, OpBuilder &builder) {
    std::array<Value, 3> result;

    auto constantOne = builder.createOrFold<mlir::ConstantIndexOp>(loc, 1);
    if (workload.size() <= 3) {
      // 1-D to 3-D are easy (pad 2 to 0 dimensions) and divide by workgroup
      // size.
      for (int i = 0; i < 3; ++i) {
        // Round up: (workload[i] + workgroup_size - 1) / workgroup_size;
        Value workloadI = i < workload.size() ? workload[i] : constantOne;
        workloadI = builder.createOrFold<mlir::SubIOp>(
            loc,
            builder.createOrFold<mlir::AddIOp>(loc, workloadI,
                                               workgroupSize[i]),
            constantOne);
        result[i] = builder.createOrFold<UnsignedDivIOp>(loc, workloadI,
                                                         workgroupSize[i]);
      }
    } else {
      // TODO(#4140): remapping of N-D to 3-D: this is not how you do this!
      Value flatWorkload = constantOne;
      for (auto workloadI : workload) {
        flatWorkload =
            builder.createOrFold<MulIOp>(loc, flatWorkload, workloadI);
      }
      for (int i = 0; i < 3; ++i) {
        // Round up: (workload[i] + workgroup_size - 1) / workgroup_size;
        auto rounded = builder.createOrFold<mlir::SubIOp>(
            loc,
            builder.createOrFold<mlir::AddIOp>(loc, flatWorkload,
                                               workgroupSize[i]),
            constantOne);
        auto workgroupCountI = builder.createOrFold<mlir::UnsignedDivIOp>(
            loc, rounded, workgroupSize[i]);
        result[i] = workgroupCountI;

        // Multiply back out and subtract from invocations.
        flatWorkload = builder.createOrFold<SubIOp>(
            loc, flatWorkload,
            builder.createOrFold<MulIOp>(loc, workgroupCountI, rounded));
      }
    }

    return result;
  }
};

static void insertSerializationBarriers(Location loc, Block &block,
                                        Value commandBuffer,
                                        OpBuilder &builder) {
  // TODO(benvanik): derive based on the type of operations that surround the
  // barriers. Can use deriveCommandCategories on the ranges to see what kind
  // of ops happen above and below, but really some analysis is required.
  auto sourceStage = IREE::HAL::ExecutionStageBitfield::CommandRetire |
                     IREE::HAL::ExecutionStageBitfield::Transfer |
                     IREE::HAL::ExecutionStageBitfield::Dispatch;
  auto targetStage = IREE::HAL::ExecutionStageBitfield::CommandIssue |
                     IREE::HAL::ExecutionStageBitfield::Transfer |
                     IREE::HAL::ExecutionStageBitfield::Dispatch;
  auto flags = IREE::HAL::ExecutionBarrierFlagBitfield::None;

  // Insert barriers after every op.
  // Note that we can't mutate the block while iterating it so we first grab
  // all the original ops.
  SmallVector<Operation *> serialOps;
  for (auto &op : block) serialOps.push_back(&op);
  for (auto *op : serialOps) {
    if (op->hasTrait<OpTrait::IsTerminator>()) continue;
    builder.setInsertionPointAfter(op);
    builder.create<IREE::HAL::CommandBufferExecutionBarrierOp>(
        loc, commandBuffer, sourceStage, targetStage, flags);
  }
}

struct CmdExecuteOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdExecuteOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdExecuteOp executeOp,
      IREE::Stream::CmdExecuteOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = executeOp.getLoc();
    auto device = lookupDeviceFor(executeOp, rewriter);

    // TODO(benvanik): disable inline execution once we have semaphores.
    // We can look ahead to see if there's an await immediately to trigger the
    // inline execution.
    auto modes = IREE::HAL::CommandBufferModeBitfield::OneShot |
                 IREE::HAL::CommandBufferModeBitfield::AllowInlineExecution;

    // Derive the command buffer type based on the kind of operations present.
    // This can help the submission get routed to appropriate hardware queues
    // (like dedicated DMA controllers).
    auto commandCategories = deriveCommandCategories(executeOp.body());

    // Create a new command buffer for recording. If we were
    auto commandBuffer =
        rewriter
            .create<IREE::HAL::CommandBufferCreateOp>(
                loc, rewriter.getType<IREE::HAL::CommandBufferType>(), device,
                modes, commandCategories)
            .result();
    mapping->mapCommandBuffer(executeOp, commandBuffer);

    // Run through the execution region and serialize execution by inserting
    // barriers. Nested regions may elide barriers as needed.
    auto &bodyBlock = executeOp.body().front();
    insertSerializationBarriers(loc, bodyBlock, commandBuffer,
                                OpBuilder::atBlockBegin(&bodyBlock));

    // Begin/end recording and inline the execution region between them.
    rewriter.create<IREE::HAL::CommandBufferBeginOp>(loc, commandBuffer);
    auto endOp =
        rewriter.create<IREE::HAL::CommandBufferEndOp>(loc, commandBuffer);
    rewriter.mergeBlockBefore(&executeOp.body().front(), endOp,
                              operands.operands());

    // TODO(benvanik): we should queue a submit here with the semaphore instead.
    rewriter.create<IREE::HAL::ExSubmitAndWaitOp>(loc, device, commandBuffer);

    // TODO(benvanik): propagate semaphore information.
    auto resolvedTimepoint =
        rewriter.create<mlir::ConstantIndexOp>(loc, 0).getResult();

    rewriter.replaceOp(executeOp, resolvedTimepoint);
    return success();
  }
};

struct CmdSerialOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdSerialOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdSerialOp serialOp,
      IREE::Stream::CmdSerialOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    auto commandBuffer = mapping->lookupCommandBufferFor(serialOp);

    // Run through the execution region and serialize execution by inserting
    // barriers. Nested regions may elide barriers as needed.
    auto &bodyBlock = serialOp.body().front();
    insertSerializationBarriers(serialOp.getLoc(), bodyBlock, commandBuffer,
                                OpBuilder::atBlockBegin(&bodyBlock));

    // Inline the serial execution region.
    rewriter.mergeBlockBefore(&serialOp.body().front(), serialOp);
    rewriter.eraseOp(serialOp);
    return success();
  }
};

struct CmdConcurrentOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdConcurrentOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdConcurrentOp concurrentOp,
      IREE::Stream::CmdConcurrentOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    // Inline the concurrent execution region.
    // TODO(benvanik): split barriers (event set/wait) when nesting.
    rewriter.mergeBlockBefore(&concurrentOp.body().front(), concurrentOp);
    rewriter.eraseOp(concurrentOp);
    return success();
  }
};

struct TimepointImmediateOpPattern
    : public StreamConversionPattern<IREE::Stream::TimepointImmediateOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::TimepointImmediateOp immediateOp,
      IREE::Stream::TimepointImmediateOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): model timepoints as semaphores.
    rewriter.replaceOpWithNewOp<mlir::ConstantIndexOp>(immediateOp, 0);
    return success();
  }
};

struct TimepointJoinOpPattern
    : public StreamConversionPattern<IREE::Stream::TimepointJoinOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::TimepointJoinOp joinOp,
      IREE::Stream::TimepointJoinOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): model timepoints as semaphores.
    // This should be a max() of the operand timepoints. MLIR has no max op
    // until https://reviews.llvm.org/D110540. Could be done with affine
    // expressions, but since everything is always 0 we just max(0,0)=0 here :)
    rewriter.replaceOpWithNewOp<mlir::ConstantIndexOp>(joinOp, 0);
    return success();
  }
};

struct TimepointAwaitOpPattern
    : public StreamConversionPattern<IREE::Stream::TimepointAwaitOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::TimepointAwaitOp awaitOp,
      IREE::Stream::TimepointAwaitOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): model timepoints as semaphores.
    rewriter.replaceOp(awaitOp, operands.operands());
    return success();
  }
};

struct ElideYieldOpPattern
    : public StreamConversionPattern<IREE::Stream::YieldOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::YieldOp yieldOp, IREE::Stream::YieldOp::Adaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(yieldOp);
    return success();
  }
};

// Annoying we have to have this here, but there's no attribute converter
// equivalent we have access to so that we could do it in a generic way.
struct GlobalTimepointConversionPattern
    : public OpConversionPattern<IREE::Util::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::GlobalOp op, llvm::ArrayRef<Value> rawOperands,
      ConversionPatternRewriter &rewriter) const override {
    auto initialValue = op.initial_value();
    if (!initialValue.hasValue()) return failure();
    if (!initialValue->isa<IREE::Stream::TimepointAttr>()) return failure();
    rewriter.updateRootInPlace(
        op, [&]() { op.initial_valueAttr(rewriter.getIndexAttr(0)); });
    return success();
  }
};

}  // namespace

void populateStreamToHALPatterns(MLIRContext *context,
                                 ConversionTarget &conversionTarget,
                                 TypeConverter &typeConverter,
                                 OwningRewritePatternList &patterns) {
  conversionTarget.addIllegalDialect<IREE::Stream::StreamDialect>();

  typeConverter.addConversion(
      [=](IREE::Stream::ResourceType type, SmallVectorImpl<Type> &results) {
        // Resources are just buffers (no shape/encoding/etc).
        results.push_back(IREE::HAL::BufferType::get(context));
        return success();
      });

  typeConverter.addConversion(
      [=](IREE::Stream::TimepointType type, SmallVectorImpl<Type> &results) {
        // TODO(benvanik): model timepoints as semaphores.
        // This may become a !hal.semaphore + index, or some !hal.timepoint that
        // we then do more analysis on once we know what devices are in use
        // where.
        results.push_back(IndexType::get(context));
        return success();
      });

  // Spooky action at a distance:
  patterns.insert<GlobalTimepointConversionPattern>(typeConverter, context);

  auto mapping = std::make_shared<StreamConversionMapping>();
  patterns.insert<ResourceAllocOpPattern, ResourceAllocaOpPattern,
                  ResourceDeallocaOpPattern, ResourceSizeOpPattern,
                  ResourceMapOpPattern, ResourceTryMapOpPattern,
                  ResourceLoadOpPattern, ResourceStoreOpPattern,
                  ResourceSubviewOpPattern>(mapping, typeConverter, context);
  patterns.insert<TensorImportOpPattern, TensorExportOpPattern>(
      mapping, typeConverter, context);
  patterns
      .insert<CmdFlushOpPattern, CmdInvalidateOpPattern, CmdDiscardOpPattern,
              CmdFillOpPattern, CmdCopyOpPattern, CmdDispatchOpPattern,
              CmdExecuteOpPattern, CmdSerialOpPattern, CmdConcurrentOpPattern>(
          mapping, typeConverter, context);
  patterns.insert<TimepointImmediateOpPattern, TimepointJoinOpPattern,
                  TimepointAwaitOpPattern>(mapping, typeConverter, context);
  patterns.insert<ElideYieldOpPattern>(mapping, typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir
