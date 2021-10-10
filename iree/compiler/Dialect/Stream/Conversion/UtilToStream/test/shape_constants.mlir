// RUN: iree-opt -split-input-file -iree-stream-conversion %s | IreeFileCheck %s

// CHECK-LABEL: @dynamic_shape_constant
func @dynamic_shape_constant() {
  // CHECK: xxx
  %c = util.dynamic_shape_constant dense<2> : tensor<2xi32> -> tensor<?xi32>
  return
}
