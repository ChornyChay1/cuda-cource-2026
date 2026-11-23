#include "utils.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

// Аргументы идут парами «ключ значение»; одиночный --help обработан в main.
ProgramOptions parse_arguments(int argc, char **argv) {
    ProgramOptions options;

    for (int idx = 1; idx < argc; idx += 2) {
        const std::string key = argv[idx];

        if (idx + 1 == argc) {
            throw std::invalid_argument("Missing value: " + key);
        }

        const std::string value = argv[idx + 1];

        if (key == "--input") {
            options.input_path = value;
        } else if (key == "--output") {
            options.output_path = value;
        } else if (key == "--experiments") {
            options.experiments_path = value;
        } else if (key == "--mode") {
            if (value == "coalesced") {
                options.noncoalesced = false;
            } else if (value == "noncoalesced") {
                options.noncoalesced = true;
            } else {
                throw std::invalid_argument("Invalid mode: " + value);
            }
        } else {
            throw std::invalid_argument("Invalid option: " + key);
        }
    }

    // Путь к экспериментам необязателен, пути входа и выхода обязательны.
    if (options.input_path.empty() || options.output_path.empty()) {
        throw std::invalid_argument("--input and --output required");
    }

    return options;
}

VectorInput read_input(const std::string &path) {
    std::ifstream input(path);
    VectorInput data;
    std::string size_token;

    if (!(input >> size_token >> data.alpha) || !std::isfinite(data.alpha)) {
        throw std::invalid_argument("Invalid input header: " + path);
    }

    // Проверяем текстовую запись N целиком, чтобы не принять 1.5 за целое 1.
    size_t parsed = 0;
    long long n;

    try {
        n = std::stoll(size_token, &parsed);
    } catch (const std::exception &) {
        throw std::invalid_argument("Invalid N");
    }

    // Проверяем весь токен и защищаем расчёт размера трёх выходных векторов от переполнения.
    if (parsed != size_token.size() || n < 0 ||
        static_cast<unsigned long long>(n) >
            std::numeric_limits<size_t>::max() / (3 * sizeof(float))) {
        throw std::invalid_argument("Invalid N");
    }

    data.x.resize(n);
    data.y.resize(n);

    // Ожидаем ровно N пар конечных чисел; NaN и бесконечности не допускаем.
    for (size_t idx = 0; idx < data.x.size(); ++idx) {
        if (!(input >> data.x[idx] >> data.y[idx]) || !std::isfinite(data.x[idx]) ||
            !std::isfinite(data.y[idx])) {
            throw std::invalid_argument("Invalid input pair " + std::to_string(idx));
        }
    }

    // После N пар разрешены только пробелы и конец файла.
    std::string extra;

    if (input >> extra) {
        throw std::invalid_argument("Unexpected input data");
    }

    return data;
}

void write_output(const std::string &path, const VectorResults &results) {
    std::ofstream output(path);
    // Девяти значащих цифр достаточно для обратного чтения FP32 без потери точности.
    output << std::setprecision(9);

    for (size_t idx = 0; idx < results.sum.size(); ++idx) {
        output << results.sum[idx] << ' ' << results.saxpy[idx] << ' ' << results.relu[idx] << '\n';
    }

    // Ошибка записи может обнаружиться и при закрытии файла.
    output.close();

    if (!output) {
        throw std::runtime_error("Cannot write output: " + path);
    }
}

std::vector<Experiment> read_experiments(const std::string &path) {
    std::ifstream input(path);
    std::vector<Experiment> experiments;
    Experiment settings{};

    while (true) {
        // Пропускаем пробелы: обычный конец файла отличается от неполной настройки.
        input >> std::ws;

        if (input.eof()) {
            break;
        }

        // Формат одной настройки: размер блока, число прогревов, число измерений.
        if (!(input >> settings.block_size >> settings.warmup >> settings.repeats)) {
            throw std::invalid_argument("Invalid experiment");
        }

        // Берём целое число групп по 32 потока; для статистики нужен хотя бы один повтор.
        if (settings.block_size < 32 || settings.block_size > 1024 ||
            settings.block_size % 32 != 0 || settings.warmup < 0 || settings.repeats < 1) {
            throw std::invalid_argument("Invalid experiment settings");
        }

        experiments.push_back(settings);
    }

    if (!input.eof() || experiments.empty()) {
        throw std::invalid_argument("Invalid experiments file: " + path);
    }

    return experiments;
}

void print_help() {
    std::cout << "submission --input FILE --output FILE [--mode coalesced|noncoalesced]\n"
                 "[--experiments FILE] | --help\n"
                 "Input: N alpha, then N float pairs x y. Default mode: coalesced.\n"
                 "Output: x+y alpha*x+y max(0,x), one row per pair; empty for N=0.\n"
                 "Experiments: rows of block_size warmup repeats; block is 32..1024, step 32,\n"
                 "warmup >= 0, repeats >= 1. Default: 256 0 1.\n";
}

// Успешный вызов ничего не печатает; ошибку передаём в обработчик через исключение.
void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    // Монотонные часы CPU подходят для измерения интервала независимо от системного времени.
    const auto finish = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> duration = finish - start;

    return duration.count();
}

double event_elapsed_ms(cudaEvent_t start, cudaEvent_t finish) {
    // События измеряют время в очереди GPU; завершение finish проверяет вызывающий код.
    float ms;
    check_cuda(cudaEventElapsedTime(&ms, start, finish));

    return ms;
}

void print_statistics(const char *name, const std::vector<double> &samples) {
    double mean = 0.0;

    for (double ms : samples) {
        mean += ms / samples.size();
    }

    // Разброс считаем по всем измерениям: делим сумму квадратов отклонений на N.
    double variance = 0.0;

    for (double ms : samples) {
        variance += (ms - mean) * (ms - mean) / samples.size();
    }

    std::cout << name << "_ms=" << mean << " stddev=" << std::sqrt(variance) << '\n';
}
