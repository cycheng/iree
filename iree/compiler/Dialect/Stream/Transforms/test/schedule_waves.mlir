// RUN: iree-opt -split-input-file -iree-stream-schedule-waves %s | IreeFileCheck %s

// CHECK: DO NOT SUBMIT
