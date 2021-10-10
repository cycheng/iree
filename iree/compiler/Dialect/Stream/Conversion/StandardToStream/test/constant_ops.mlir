// RUN: iree-opt -split-input-file -iree-stream-conversion %s | IreeFileCheck %s

// CHECK-LABEL: @constantTensor
func @constantTensor() {
  // CHECK: xx
  %0 = constant dense<[1, 2]> : tensor<2xi32>
  return
}

// -----

// CHECK-LABEL: @constantTensor1
func @constantTensor1() {
  // CHECK: xx
  %0 = constant dense<[1, 0]> : tensor<2xi1>
  return
}
