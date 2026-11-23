#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <string>
#include <vector>

// Пути и режим, полученные из аргументов командной строки.
struct ProgramOptions {
    std::string input_path;
    std::string output_path;
    std::string experiments_path;
    bool noncoalesced = false;
};

// Исходные данные в памяти CPU. alpha — коэффициент операции SAXPY.
struct VectorInput {
    std::vector<float> x;
    std::vector<float> y;
    float alpha = 0.0f;
};

// Три независимых результата в памяти CPU, в исходном порядке элементов.
struct VectorResults {
    std::vector<float> sum;
    std::vector<float> saxpy;
    std::vector<float> relu;
};

// Одна настройка из файла: потоки в блоке, прогревочные и измеряемые повторы.
struct Experiment {
    int block_size;
    int warmup;
    int repeats;
};

// Разбор аргументов и файлов; неверные данные приводят к исключению invalid_argument.
ProgramOptions parse_arguments(int argc, char **argv);
VectorInput read_input(const std::string &path);
void write_output(const std::string &path, const VectorResults &results);
std::vector<Experiment> read_experiments(const std::string &path);
// Краткое описание CLI, режимов и форматов файлов.
void print_help();

void check_cuda(cudaError_t status);

// Время CPU с момента start, в миллисекундах.
double elapsed_ms(std::chrono::steady_clock::time_point start);

// Время между событиями GPU, в миллисекундах. Событие finish должно завершиться.
double event_elapsed_ms(cudaEvent_t start, cudaEvent_t finish);
// Выводит среднее и стандартное отклонение измерений в миллисекундах.
void print_statistics(const char *name, const std::vector<double> &samples);

// Время одного цикла в миллисекундах: этапы измерены на GPU, total — таймером CPU.
struct RunTimes {
    double h2d = 0.0;
    double kernels = 0.0;
    double d2h = 0.0;
    double total = 0.0;
};
