#pragma once

#include <cuda_runtime.h>

// __global__ обозначает ядро: его запускают с CPU, а выполняют потоки GPU.
// Все указатели ниже относятся к памяти GPU; n — число элементов.
// noncoalesced выбирает доступ с шагом 32 float вместо соседних адресов.
__global__ void vector_add_kernel(const float *x, const float *y, float *sum, size_t n,
                                  bool noncoalesced);

__global__ void saxpy_kernel(const float *x, const float *y, float *saxpy, size_t n, float alpha,
                             bool noncoalesced);

__global__ void relu_kernel(const float *x, float *relu, size_t n, bool noncoalesced);
