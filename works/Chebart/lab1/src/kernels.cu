#include "kernels.h"

// В каждой логической плитке 32 строки и 32 столбца, всего 1024 элемента.
constexpr size_t TILE_WIDTH = 32;
constexpr size_t TILE_SIZE = TILE_WIDTH * TILE_WIDTH;

// В режиме noncoalesced переставляем индексы внутри плитки 32 x 32.
// Соседние потоки получают элементы с шагом 32 float вместо соседних элементов.
__device__ size_t element_index(size_t logical_idx, bool noncoalesced) {
    if (noncoalesced) {
        const size_t tile_start = logical_idx / TILE_SIZE * TILE_SIZE;
        const size_t local_idx = logical_idx % TILE_SIZE;

        // Остаток даёт столбец, деление — строку. Меняем их местами, как при транспонировании.
        return tile_start + (local_idx % TILE_WIDTH) * TILE_WIDTH + local_idx / TILE_WIDTH;
    }

    // Для coalesced сохраняем порядок: соседние потоки получают соседние элементы.
    return logical_idx;
}

// Каждый поток независимо вычисляет сложение для назначенных ему элементов.
__global__ void vector_add_kernel(const float *x, const float *y, float *sum, size_t n,
                                  bool noncoalesced) {
    // blockIdx.x — номер блока, blockDim.x — потоков в блоке, threadIdx.x — номер потока.
    const size_t start_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const size_t padded_n = (n + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;

    // Шаг равен общему числу потоков сетки: один поток может обработать несколько элементов.
    // Проверяем границу после перестановки, чтобы не потерять хвост последней плитки.
    for (size_t logical_idx = start_idx; logical_idx < padded_n; logical_idx += stride) {
        const size_t idx = element_index(logical_idx, noncoalesced);

        if (idx < n) {
            sum[idx] = x[idx] + y[idx];
        }
    }
}

// SAXPY: alpha * x + y. Исходный y не изменяем, результат пишем отдельно.
__global__ void saxpy_kernel(const float *x, const float *y, float *saxpy, size_t n, float alpha,
                             bool noncoalesced) {
    const size_t start_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const size_t padded_n = (n + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;

    for (size_t logical_idx = start_idx; logical_idx < padded_n; logical_idx += stride) {
        const size_t idx = element_index(logical_idx, noncoalesced);

        if (idx < n) {
            // Умножение и сложение округляются отдельно в FP32, как в эталоне NumPy.
            saxpy[idx] = __fadd_rn(__fmul_rn(alpha, x[idx]), y[idx]);
        }
    }
}

// ReLU заменяет отрицательные значения x на ноль, остальные сохраняет.
__global__ void relu_kernel(const float *x, float *relu, size_t n, bool noncoalesced) {
    const size_t start_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const size_t padded_n = (n + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;

    for (size_t logical_idx = start_idx; logical_idx < padded_n; logical_idx += stride) {
        const size_t idx = element_index(logical_idx, noncoalesced);

        if (idx < n) {
            relu[idx] = fmaxf(0.0f, x[idx]);
        }
    }
}
