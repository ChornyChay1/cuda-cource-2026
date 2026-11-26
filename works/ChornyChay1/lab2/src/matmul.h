#pragma once

#include <cuda_runtime.h>

#include <cstddef>

struct Shape {
  int m;
  int k;
  int n;
};

void CheckCuda(cudaError_t status);
void LaunchMatmul(const float* a, const float* b, float* c, Shape shape,
                  bool tiled, int tile);
cudaFuncAttributes MatmulAttributes(bool tiled);

struct DeviceMatrices {
  float* a = nullptr;
  float* b = nullptr;
  float* c = nullptr;

  DeviceMatrices() = default;
  DeviceMatrices(const DeviceMatrices&) = delete;
  DeviceMatrices& operator=(const DeviceMatrices&) = delete;
  ~DeviceMatrices();
  void Upload(Shape shape, const float* host_a, const float* host_b);
};
