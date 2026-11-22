#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr unsigned warpSizeForLayout = 32;
constexpr int warmups = 3;
using Clock = std::chrono::steady_clock;

class InputError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct Options {
    std::string input;
    std::string output;
    bool strided = false;
    unsigned blockSize = 256;
    int repeats = 0;
};

void printHelp() {
    std::cout << "Usage: submission --input PATH --output PATH "
                 "[--mode coalesced|noncoalesced] [--block-size 128|256|512] "
                 "[--repeats N]\n"
                 "Input: N alpha followed by N pairs x y (FP32).\n"
                 "Output: x+y, alpha*x+y, max(0,x), one row per input pair.\n"
                 "Default mode: coalesced. Default block size: 256.\n"
                 "With --repeats N (N > 0), report median timings after three warmups.\n"
                 "For N=0, write an empty output file.\n";
}

template <typename Integer> Integer parseInteger(std::string_view value, const char *message) {
    if (!value.empty() && value.front() == '+') {
        value.remove_prefix(1);
    }
    Integer result{};
    const char *end = value.data() + value.size();
    const auto parsed = std::from_chars(value.data(), end, result);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
        throw InputError(message);
    }
    return result;
}

Options parseOptions(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (++i == argc || std::string_view(argv[i]).empty() ||
            std::string_view(argv[i]).substr(0, 2) == "--") {
            throw InputError("Missing value for " + std::string(flag));
        }
        const std::string_view value = argv[i];

        if (flag == "--input") {
            options.input = value;
        } else if (flag == "--output") {
            options.output = value;
        } else if (flag == "--mode") {
            if (value != "coalesced" && value != "noncoalesced") {
                throw InputError("Mode must be coalesced or noncoalesced");
            }
            options.strided = value == "noncoalesced";
        } else if (flag == "--block-size") {
            options.blockSize = parseInteger<unsigned>(value, "Invalid block size");
            if (options.blockSize != 128 && options.blockSize != 256 && options.blockSize != 512) {
                throw InputError("Block size must be 128, 256, or 512");
            }
        } else if (flag == "--repeats") {
            options.repeats = parseInteger<int>(value, "Invalid repeat count");
            if (options.repeats <= 0) {
                throw InputError("Repeat count must be positive");
            }
        } else {
            throw InputError("Unknown option: " + std::string(flag));
        }
    }
    if (options.input.empty() || options.output.empty()) {
        throw InputError("--input and --output are required");
    }
    return options;
}

struct Input {
    std::size_t count;
    float alpha;
    std::vector<float> pairs;
};

bool readFloat(std::istream &file, std::string &token, float &value) {
    if (!(file >> token)) {
        return false;
    }
    char *end = nullptr;
    const char *start = token.c_str();
    value = std::strtof(start, &end);
    return end == start + token.size() && std::isfinite(value);
}

Input readInput(const std::string &path) {
    std::ifstream file(path);
    std::string countText;
    std::string token;
    Input input{};
    if (!(file >> countText) || !readFloat(file, token, input.alpha)) {
        throw InputError("Cannot read N and alpha from " + path);
    }
    input.count = parseInteger<std::size_t>(countText, "N must be a nonnegative integer");
    if (input.count > std::numeric_limits<std::size_t>::max() / (3 * sizeof(float))) {
        throw InputError("N is too large");
    }
    for (std::size_t i = 0; i < input.count; ++i) {
        float x, y;
        if (!readFloat(file, token, x) || !readFloat(file, token, y)) {
            throw InputError("Invalid or missing input pair " + std::to_string(i + 1));
        }
        input.pairs.push_back(x);
        input.pairs.push_back(y);
    }
    if (file >> token || file.bad()) {
        throw InputError("Expected exactly N input pairs");
    }
    return input;
}

void writeOutput(const std::string &path, const float *output, std::size_t count) {
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open output file " + path);
    }
    file << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (std::size_t i = 0; i < count; ++i) {
        file << output[i] << ' ' << output[count + i] << ' ' << output[2 * count + i] << '\n';
    }
    file.close();
    if (!file) {
        throw std::runtime_error("Cannot write output file " + path);
    }
}

void checkCuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct FreeDevice {
    void operator()(float *pointer) const { cudaFree(pointer); }
};

struct FreeHost {
    void operator()(float *pointer) const { cudaFreeHost(pointer); }
};

class Event {
  public:
    Event() { checkCuda(cudaEventCreate(&handle), "cudaEventCreate"); }
    ~Event() { cudaEventDestroy(handle); }

    Event(const Event &) = delete;
    Event &operator=(const Event &) = delete;

    cudaEvent_t handle{};
};

template <bool strided>
__global__ void vectorOperations(const float *input, float *output, std::size_t count,
                                 float alpha) {
    const unsigned thread = threadIdx.x;
    unsigned local = thread;
    if constexpr (strided) {
        local = (thread % warpSizeForLayout) * (blockDim.x / warpSizeForLayout) +
                thread / warpSizeForLayout;
    }
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + local;
    if (i < count) {
        const float x = input[i];
        const float y = input[count + i];
        output[i] = x + y;
        output[count + i] = fmaf(alpha, x, y);
        output[2 * count + i] = fmaxf(0.0f, x);
    }
}

struct Timing {
    double h2d;
    double kernel;
    double d2h;
    double total;
};

double millisecondsSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

float elapsed(const Event &first, const Event &last) {
    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, first.handle, last.handle),
              "cudaEventElapsedTime");
    return milliseconds;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return (values[(values.size() - 1) / 2] + values[middle]) / 2;
}

void printTimings(const std::vector<Timing> &samples, double allocation, const Options &options,
                  std::size_t count) {
    std::vector<double> h2d, kernel, d2h, total;
    for (const auto &sample : samples) {
        h2d.push_back(sample.h2d);
        kernel.push_back(sample.kernel);
        d2h.push_back(sample.d2h);
        total.push_back(sample.total);
    }
    std::cout << std::fixed << std::setprecision(6) << "N=" << count
              << " mode=" << (options.strided ? "noncoalesced" : "coalesced")
              << " block=" << options.blockSize << " repeats=" << samples.size()
              << " warmups=" << warmups << " H2D_ms=" << median(h2d)
              << " kernel_ms=" << median(kernel) << " D2H_ms=" << median(d2h)
              << " total_ms=" << median(total) << " allocation_ms=" << allocation << '\n';
}

void run(const Options &options, const Input &input) {
    const std::size_t count = input.count;
    if (count == 0) {
        writeOutput(options.output, nullptr, 0);
        return;
    }

    const auto blocks = 1 + (count - 1) / options.blockSize;
    int device = 0;
    cudaDeviceProp properties{};
    checkCuda(cudaGetDevice(&device), "cudaGetDevice");
    checkCuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    if (blocks > static_cast<unsigned>(properties.maxGridSize[0]) ||
        options.blockSize > static_cast<unsigned>(properties.maxThreadsPerBlock)) {
        throw InputError("Input size or block size exceeds GPU launch limits");
    }
    const auto gridSize = static_cast<unsigned>(blocks);
    const std::size_t inputBytes = 2 * count * sizeof(float);
    const std::size_t outputBytes = 3 * count * sizeof(float);

    const auto allocationStart = Clock::now();
    float *pointer = nullptr;
    checkCuda(cudaMallocHost(&pointer, inputBytes), "cudaMallocHost(input)");
    std::unique_ptr<float, FreeHost> hostInput(pointer);
    checkCuda(cudaMallocHost(&pointer, outputBytes), "cudaMallocHost(output)");
    std::unique_ptr<float, FreeHost> hostOutput(pointer);
    checkCuda(cudaMalloc(&pointer, inputBytes), "cudaMalloc(input)");
    std::unique_ptr<float, FreeDevice> deviceInput(pointer);
    checkCuda(cudaMalloc(&pointer, outputBytes), "cudaMalloc(output)");
    std::unique_ptr<float, FreeDevice> deviceOutput(pointer);
    const double allocation = millisecondsSince(allocationStart);

    for (std::size_t i = 0; i < count; ++i) {
        hostInput.get()[i] = input.pairs[2 * i];
        hostInput.get()[count + i] = input.pairs[2 * i + 1];
    }

    const auto execute = [&](const std::array<Event, 4> *events) {
        const auto begin = Clock::now();
        const auto record = [&](std::size_t index) {
            if (events != nullptr) {
                checkCuda(cudaEventRecord((*events)[index].handle), "cudaEventRecord");
            }
        };

        record(0);
        checkCuda(
            cudaMemcpyAsync(deviceInput.get(), hostInput.get(), inputBytes, cudaMemcpyHostToDevice),
            "cudaMemcpyAsync(H2D)");
        record(1);
        if (options.strided) {
            vectorOperations<true><<<gridSize, options.blockSize>>>(
                deviceInput.get(), deviceOutput.get(), count, input.alpha);
        } else {
            vectorOperations<false><<<gridSize, options.blockSize>>>(
                deviceInput.get(), deviceOutput.get(), count, input.alpha);
        }
        checkCuda(cudaGetLastError(), "vectorOperations launch");
        record(2);
        checkCuda(cudaMemcpyAsync(hostOutput.get(), deviceOutput.get(), outputBytes,
                                  cudaMemcpyDeviceToHost),
                  "cudaMemcpyAsync(D2H)");
        record(3);
        checkCuda(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize");
        const double total = millisecondsSince(begin);
        if (events == nullptr) {
            return Timing{};
        }
        return Timing{elapsed((*events)[0], (*events)[1]), elapsed((*events)[1], (*events)[2]),
                      elapsed((*events)[2], (*events)[3]), total};
    };

    if (options.repeats == 0) {
        execute(nullptr);
    } else {
        for (int i = 0; i < warmups; ++i) {
            execute(nullptr);
        }
        const std::array<Event, 4> events{};
        std::vector<Timing> samples;
        for (int i = 0; i < options.repeats; ++i) {
            samples.push_back(execute(&events));
        }
        printTimings(samples, allocation, options, count);
    }
    writeOutput(options.output, hostOutput.get(), count);
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            printHelp();
            return 0;
        }
        const Options options = parseOptions(argc, argv);
        run(options, readInput(options.input));
        return 0;
    } catch (const InputError &error) {
        std::cerr << error.what() << '\n';
        return 2;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
