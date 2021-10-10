// RUN: iree-opt -split-input-file -iree-conver-to-stream-pass %s | IreeFileCheck %s

func @static_tensor_cast_to_dynamic(%arg0: tensor<4x4xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[C4:.*]] = constant 4 : index
  // CHECK-DAG: %[[RESULT:.*]] = flow.tensor.reshape %arg0 : tensor<4x4xf32> -> tensor<?x?xf32>{%[[C4]], %[[C4]]}
  // CHECK: return %[[RESULT]]
  %0 = tensor.cast %arg0 : tensor<4x4xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
