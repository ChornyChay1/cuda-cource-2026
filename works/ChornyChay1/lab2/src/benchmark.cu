#include <cublas_v2.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "matmul.h"

namespace {

constexpr int kWarmups = 3;
constexpr int kRepeats = 30;
constexpr int kBankBlocks = 256;
constexpr int kBankThreads = 256;
constexpr int kBankReads = 256;

void CheckBlas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cuBLAS error " + std::to_string(status));
  }
}

struct Stopwatch {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  ~Stopwatch() {
    if (stop != nullptr) {
      cudaEventDestroy(stop);
    }
    if (start != nullptr) {
      cudaEventDestroy(start);
    }
  }

  void Create() {
    CheckCuda(cudaEventCreate(&start));
    CheckCuda(cudaEventCreate(&stop));
  }

  void Begin() { CheckCuda(cudaEventRecord(start)); }

  float End() {
    CheckCuda(cudaEventRecord(stop));
    CheckCuda(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

struct Blas {
  cublasHandle_t handle = nullptr;

  ~Blas() {
    if (handle != nullptr) {
      cublasDestroy(handle);
    }
  }

  void Multiply(const DeviceMatrices& gpu, Shape shape) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // The row-major buffers look transposed to cuBLAS. Compute C^T = B^T A^T.
    CheckBlas(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, shape.n, shape.m, shape.k, &alpha,
        gpu.b, CUDA_R_32F, shape.n, gpu.a, CUDA_R_32F, shape.k, &beta, gpu.c,
        CUDA_R_32F, shape.n, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
  }
};

float Median(std::vector<float>& samples) {
  std::sort(samples.begin(), samples.end());
  return (samples[kRepeats / 2 - 1] + samples[kRepeats / 2]) / 2;
}

void RunMatmul(const DeviceMatrices& gpu, Shape shape, int tile, bool tiled,
               Blas* blas) {
  if (blas != nullptr) {
    blas->Multiply(gpu, shape);
  } else {
    LaunchMatmul(gpu.a, gpu.b, gpu.c, shape, tiled, tile);
  }
}

float TimeMatmul(const DeviceMatrices& gpu, Shape shape, int tile, bool tiled,
                 Blas* blas, Stopwatch& timer) {
  for (int i = 0; i < kWarmups; ++i) {
    RunMatmul(gpu, shape, tile, tiled, blas);
  }
  std::vector<float> samples;
  for (int i = 0; i < kRepeats; ++i) {
    timer.Begin();
    RunMatmul(gpu, shape, tile, tiled, blas);
    samples.push_back(timer.End());
  }
  return Median(samples);
}

void PrintMatmul(Shape shape, const char* mode, int tile, float ms) {
  const double operations = 2.0 * shape.m * shape.k * shape.n;
  std::cout << mode << " M=" << shape.m << " K=" << shape.k << " N=" << shape.n
            << " tile=" << tile << " ms=" << ms
            << " GFLOP/s=" << operations / (ms * 1e6);
  if (tile > 0) {
    const bool tiled = std::string(mode) == "tiled";
    const auto attributes = MatmulAttributes(tiled);
    const auto dynamic_bytes = tiled ? 2 * tile * tile * sizeof(float) : 0;
    std::cout << " registers/thread=" << attributes.numRegs
              << " shared_bytes/block="
              << attributes.sharedSizeBytes + dynamic_bytes;
  }
  std::cout << '\n';
}

void BenchmarkShape(Shape shape, Blas& blas, Stopwatch& timer) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> random(-1.0f, 1.0f);
  std::vector<float> a(shape.m * shape.k);
  std::vector<float> b(shape.k * shape.n);
  for (float& value : a) {
    value = random(rng);
  }
  for (float& value : b) {
    value = random(rng);
  }
  DeviceMatrices gpu;
  gpu.Upload(shape, a.data(), b.data());
  for (const int tile : {8, 16, 32}) {
    PrintMatmul(shape, "naive", tile,
                TimeMatmul(gpu, shape, tile, false, nullptr, timer));
    PrintMatmul(shape, "tiled", tile,
                TimeMatmul(gpu, shape, tile, true, nullptr, timer));
  }
  PrintMatmul(shape, "cuBLAS", 0,
              TimeMatmul(gpu, shape, 0, false, &blas, timer));
}

__global__ void BankReads(float* output, int stride) {
  extern __shared__ volatile float data[];
  const int t = threadIdx.x;
  for (int i = t; i < 32 * 32; i += blockDim.x) {
    data[(i / 32) * stride + i % 32] = static_cast<float>(i);
  }
  __syncthreads();

  // Within one warp all lanes read the same column of different rows.
  const int row = t % 32;
  const int warp = t / 32;
  float sum = 0.0f;
  for (int i = 0; i < kBankReads; ++i) {
    const int col = (i + warp) % 32;
    sum += data[row * stride + col];
  }
  output[blockIdx.x * blockDim.x + t] = sum;
}

void LaunchBankReads(float* output, int stride) {
  BankReads<<<kBankBlocks, kBankThreads, 32 * stride * sizeof(float)>>>(output,
                                                                        stride);
  CheckCuda(cudaGetLastError());
}

float TimeBankReads(float* output, int stride, Stopwatch& timer) {
  for (int i = 0; i < kWarmups; ++i) {
    LaunchBankReads(output, stride);
  }
  std::vector<float> samples;
  for (int i = 0; i < kRepeats; ++i) {
    timer.Begin();
    LaunchBankReads(output, stride);
    samples.push_back(timer.End());
  }
  return Median(samples);
}

std::vector<float> ReadBankOutput(const float* output) {
  std::vector<float> host(kBankBlocks * kBankThreads);
  CheckCuda(cudaMemcpy(host.data(), output, host.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));
  return host;
}

void BenchmarkBanks(Stopwatch& timer) {
  DeviceMatrices gpu;
  CheckCuda(cudaMalloc(&gpu.c, kBankBlocks * kBankThreads * sizeof(float)));
  const float compact_ms = TimeBankReads(gpu.c, 32, timer);
  const auto compact = ReadBankOutput(gpu.c);
  const float padded_ms = TimeBankReads(gpu.c, 33, timer);
  if (compact != ReadBankOutput(gpu.c)) {
    throw std::runtime_error("Bank experiment returned different sums");
  }
  std::cout << "bank stride=32 ms=" << compact_ms << '\n'
            << "bank stride=33 ms=" << padded_ms << '\n';
}

}  // namespace

int main() {
  try {
    cudaDeviceProp device{};
    int device_id = 0;
    int driver = 0;
    int runtime = 0;
    CheckCuda(cudaGetDevice(&device_id));
    CheckCuda(cudaGetDeviceProperties(&device, device_id));
    CheckCuda(cudaDriverGetVersion(&driver));
    CheckCuda(cudaRuntimeGetVersion(&runtime));
    Blas blas;
    CheckBlas(cublasCreate(&blas.handle));
    int blas_version = 0;
    CheckBlas(cublasGetVersion(blas.handle, &blas_version));
    std::cout << "GPU=" << device.name << " CUDA_driver_API=" << driver
              << " CUDA_runtime=" << runtime << " cuBLAS=" << blas_version
              << " warmups=" << kWarmups << " repeats=" << kRepeats << '\n'
              << std::fixed << std::setprecision(6);

    Stopwatch timer;
    timer.Create();
    for (const Shape shape :
         {Shape{127, 65, 257}, Shape{512, 512, 512}, Shape{1024, 1024, 1024}}) {
      BenchmarkShape(shape, blas, timer);
    }
    BenchmarkBanks(timer);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
