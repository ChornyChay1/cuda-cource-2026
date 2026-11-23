#include "kernels.h"
#include "utils.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>

// Один цикл передач и вычислений. Буферы GPU уже выделены вызывающим кодом.
// Возвращаем отдельно время этапов на GPU и полного цикла со стороны CPU.
RunTimes run_once(const VectorInput &h_input, VectorResults &h_results, float *d_x, float *d_y,
                  float *d_output, int block_size, bool noncoalesced) {
    const size_t n = h_input.x.size();
    const size_t bytes = n * sizeof(float);
    // Округляем число блоков вверх. Большие входы обрабатывает цикл внутри ядра.
    const size_t required_blocks = (n + block_size - 1) / block_size;
    const size_t grid_size = std::min<size_t>(required_blocks, 65535);
    const unsigned num_blocks = static_cast<unsigned>(grid_size);

    // Один выходной буфер содержит три вектора подряд.
    float *d_sum = d_output;
    float *d_saxpy = d_output + n;
    float *d_relu = d_output + 2 * n;

    cudaEvent_t start = nullptr;
    cudaEvent_t h2d_finished = nullptr;
    cudaEvent_t kernels_finished = nullptr;
    cudaEvent_t finish = nullptr;
    RunTimes times;
    std::string error_message;

    try {
        // Четыре события отмечают границы H2D, вычислений и D2H в очереди GPU.
        // Их создание не включаем во время полного цикла.
        check_cuda(cudaEventCreate(&start));
        check_cuda(cudaEventCreate(&h2d_finished));
        check_cuda(cudaEventCreate(&kernels_finished));
        check_cuda(cudaEventCreate(&finish));
        const auto cpu_start = std::chrono::steady_clock::now();

        // H2D: для SAXPY каждый раз передаём исходный y.
        check_cuda(cudaEventRecord(start));
        check_cuda(cudaMemcpy(d_x, h_input.x.data(), bytes, cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(d_y, h_input.y.data(), bytes, cudaMemcpyHostToDevice));
        check_cuda(cudaEventRecord(h2d_finished));

        // В <<< >>> задаём число блоков и потоков в блоке.
        // cudaGetLastError проверяет запуск; выполнение ещё может продолжаться на GPU.
        vector_add_kernel<<<num_blocks, block_size>>>(d_x, d_y, d_sum, n, noncoalesced);
        check_cuda(cudaGetLastError());

        saxpy_kernel<<<num_blocks, block_size>>>(d_x, d_y, d_saxpy, n, h_input.alpha, noncoalesced);
        check_cuda(cudaGetLastError());

        relu_kernel<<<num_blocks, block_size>>>(d_x, d_relu, n, noncoalesced);
        check_cuda(cudaGetLastError());
        check_cuda(cudaEventRecord(kernels_finished));

        // D2H: возвращаем три результата на CPU.
        check_cuda(cudaMemcpy(h_results.sum.data(), d_sum, bytes, cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(h_results.saxpy.data(), d_saxpy, bytes, cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(h_results.relu.data(), d_relu, bytes, cudaMemcpyDeviceToHost));
        check_cuda(cudaEventRecord(finish));
        // Ждём последнего события: результаты готовы, ошибки выполнения будут обнаружены.
        check_cuda(cudaEventSynchronize(finish));

        // CPU-таймер учитывает также постановку команд и ожидание GPU.
        times.total = elapsed_ms(cpu_start);
        times.h2d = event_elapsed_ms(start, h2d_finished);
        times.kernels = event_elapsed_ms(h2d_finished, kernels_finished);
        times.d2h = event_elapsed_ms(kernels_finished, finish);
    } catch (const std::exception &error) {
        // Сначала запоминаем ошибку, чтобы перед выходом успеть удалить события.
        error_message = error.what();
    }

    // Удаляем и частично созданные события; исходную ошибку не заменяем ошибкой очистки.
    cudaEvent_t events[] = {start, h2d_finished, kernels_finished, finish};

    for (cudaEvent_t event : events) {
        if (event != nullptr) {
            const cudaError_t status = cudaEventDestroy(event);

            if (status != cudaSuccess && error_message.empty()) {
                error_message = cudaGetErrorString(status);
            }
        }
    }

    if (!error_message.empty()) {
        throw std::runtime_error(error_message);
    }

    return times;
}

int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        print_help();
        return 0;
    }

    try {
        // Аргументы задают пути и режим; входной файл содержит сами векторы.
        const ProgramOptions options = parse_arguments(argc, argv);
        const VectorInput h_input = read_input(options.input_path);
        std::vector<Experiment> experiments;

        if (options.experiments_path.empty()) {
            // Обычный запуск: блок 256, без прогрева, один вычислительный цикл.
            experiments.push_back({256, 0, 1});
        } else {
            experiments = read_experiments(options.experiments_path);
        }

        VectorResults h_results;

        if (h_input.x.empty()) {
            // Для N=0 создаём пустой результат, не инициализируя GPU.
            write_output(options.output_path, h_results);
            return 0;
        }

        // Каждая строка файла параметров задаёт самостоятельный эксперимент.
        for (const Experiment &settings : experiments) {
            const size_t n = h_input.x.size();
            const size_t bytes = n * sizeof(float);
            h_results.sum.resize(n);
            h_results.saxpy.resize(n);
            h_results.relu.resize(n);

            // Этот вызов инициализирует контекст CUDA до начала измерений.
            check_cuda(cudaFree(nullptr));
            int device_id;
            check_cuda(cudaGetDevice(&device_id));

            cudaDeviceProp properties{};
            check_cuda(cudaGetDeviceProperties(&properties, device_id));

            // Допустимый в файле размер блока может превышать предел конкретной GPU.
            if (settings.block_size > properties.maxThreadsPerBlock) {
                throw std::invalid_argument("Block exceeds GPU limit");
            }

            // h_ обозначает память CPU, d_ — память GPU. nullptr означает: буфера ещё нет.
            float *d_x = nullptr;
            float *d_y = nullptr;
            float *d_output = nullptr;
            std::string error_message;
            std::vector<double> h2d_times;
            std::vector<double> kernel_times;
            std::vector<double> d2h_times;
            std::vector<double> total_times;

            try {
                // Память выделяем один раз для всех прогревов и повторов этой настройки.
                const auto allocation_start = std::chrono::steady_clock::now();
                check_cuda(cudaMalloc(&d_x, bytes));
                check_cuda(cudaMalloc(&d_y, bytes));
                check_cuda(cudaMalloc(&d_output, 3 * bytes));

                const double allocation_ms = elapsed_ms(allocation_start);
                std::cout << "block=" << settings.block_size << " warmup=" << settings.warmup
                          << " repeats=" << settings.repeats << '\n';
                std::cout << "allocation_ms=" << allocation_ms << '\n';

                // Прогрев выполняет ту же работу, но его время не сохраняем.
                for (int repeat = 0; repeat < settings.warmup; ++repeat) {
                    run_once(h_input, h_results, d_x, d_y, d_output, settings.block_size,
                             options.noncoalesced);
                }

                // Для измеряемых повторов сохраняем четыре времени отдельно.
                for (int repeat = 0; repeat < settings.repeats; ++repeat) {
                    const RunTimes times = run_once(h_input, h_results, d_x, d_y, d_output,
                                                    settings.block_size, options.noncoalesced);

                    h2d_times.push_back(times.h2d);
                    kernel_times.push_back(times.kernels);
                    d2h_times.push_back(times.d2h);
                    total_times.push_back(times.total);
                }
            } catch (const std::exception &error) {
                error_message = error.what();
            }

            // Освобождаем все уже выделенные буферы, даже если запуск завершился ошибкой.
            float *device_pointers[] = {d_x, d_y, d_output};

            for (float *pointer : device_pointers) {
                if (pointer != nullptr) {
                    const cudaError_t status = cudaFree(pointer);

                    if (status != cudaSuccess && error_message.empty()) {
                        error_message = cudaGetErrorString(status);
                    }
                }
            }

            if (!error_message.empty()) {
                throw std::runtime_error(error_message);
            }

            // В статистику попали только измеряемые повторы, без прогрева.
            print_statistics("H2D", h2d_times);
            print_statistics("kernels", kernel_times);
            print_statistics("D2H", d2h_times);
            print_statistics("total", total_times);
        }

        // Все настройки считают одно и то же: записываем результат последней.
        write_output(options.output_path, h_results);
        return 0;
    } catch (const std::invalid_argument &error) {
        std::cerr << error.what() << '\n';
        return 2;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
