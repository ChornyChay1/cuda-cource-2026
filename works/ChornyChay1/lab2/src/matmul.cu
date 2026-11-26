#include <stdexcept>

#include "matmul.h"

namespace {

__global__ void NaiveMatmul(const float* a, const float* b, float* c,
                            Shape shape) {
  const auto row =
      static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  const auto col =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < shape.m && col < shape.n) {
    float sum = 0.0f;
    for (int k = 0; k < shape.k; ++k) {
      sum = fmaf(a[row * shape.k + k],
                 b[static_cast<std::size_t>(k) * shape.n + col], sum);
    }
    c[row * shape.n + col] = sum;
  }
}

__global__ void TiledMatmul(const float* a, const float* b, float* c,
                            Shape shape) {
  extern __shared__ float shared[];
  const int tile = blockDim.x;
  float* tile_a = shared;
  float* tile_b = shared + tile * tile;
  const int x = threadIdx.x;
  const int y = threadIdx.y;
  const auto row = static_cast<std::size_t>(blockIdx.y) * tile + y;
  const auto col = static_cast<std::size_t>(blockIdx.x) * tile + x;
  const int local = y * tile + x;
  float sum = 0.0f;
  for (std::size_t start = 0; start < shape.k; start += tile) {
    const auto a_col = start + x;
    const auto b_row = start + y;
    tile_a[local] =
        (row < shape.m && a_col < shape.k) ? a[row * shape.k + a_col] : 0.0f;
    tile_b[local] =
        (b_row < shape.k && col < shape.n) ? b[b_row * shape.n + col] : 0.0f;
    __syncthreads();
    for (int k = 0; k < tile; ++k) {
      sum = fmaf(tile_a[y * tile + k], tile_b[k * tile + x], sum);
    }
    __syncthreads();
  }
  if (row < shape.m && col < shape.n) {
    c[row * shape.n + col] = sum;
  }
}

}  // namespace

void CheckCuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

void LaunchMatmul(const float* a, const float* b, float* c, Shape shape,
                  bool tiled, int tile) {
  const dim3 block(tile, tile);
  const dim3 grid(1 + (shape.n - 1) / tile, 1 + (shape.m - 1) / tile);
  if (tiled) {
    TiledMatmul<<<grid, block, 2 * tile * tile * sizeof(float)>>>(a, b, c,
                                                                  shape);
  } else {
    NaiveMatmul<<<grid, block>>>(a, b, c, shape);
  }
  CheckCuda(cudaGetLastError());
}

cudaFuncAttributes MatmulAttributes(bool tiled) {
  cudaFuncAttributes attributes{};
  if (tiled) {
    CheckCuda(cudaFuncGetAttributes(&attributes, TiledMatmul));
  } else {
    CheckCuda(cudaFuncGetAttributes(&attributes, NaiveMatmul));
  }
  return attributes;
}

DeviceMatrices::~DeviceMatrices() {
  cudaFree(c);
  cudaFree(b);
  cudaFree(a);
}

void DeviceMatrices::Upload(Shape shape, const float* host_a,
                            const float* host_b) {
  const auto a_bytes =
      static_cast<std::size_t>(shape.m) * shape.k * sizeof(float);
  const auto b_bytes =
      static_cast<std::size_t>(shape.k) * shape.n * sizeof(float);
  const auto c_bytes =
      static_cast<std::size_t>(shape.m) * shape.n * sizeof(float);
  CheckCuda(cudaMalloc(&a, a_bytes));
  CheckCuda(cudaMalloc(&b, b_bytes));
  CheckCuda(cudaMalloc(&c, c_bytes));
  CheckCuda(cudaMemcpy(a, host_a, a_bytes, cudaMemcpyHostToDevice));
  CheckCuda(cudaMemcpy(b, host_b, b_bytes, cudaMemcpyHostToDevice));
}
