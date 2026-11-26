#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "matmul.h"

namespace {

class InputError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct Options {
  std::string input;
  std::string output;
  bool tiled = true;
  int tile = 16;
};

struct Matrices {
  Shape shape{};
  std::vector<float> a;
  std::vector<float> b;
};

void PrintHelp() {
  std::cout
      << "Usage: submission --input PATH --output PATH [--mode naive|tiled]"
         " [--tile 8|16|32]\n"
         "Input: positive M K N, then row-major FP32 matrices A and B.\n"
         "Output: M N, then row-major FP32 matrix C.\n"
         "Defaults: --mode tiled --tile 16.\n";
}

int ParseTile(const std::string& value) {
  if (value == "8") {
    return 8;
  }
  if (value == "16") {
    return 16;
  }
  if (value == "32") {
    return 32;
  }
  throw InputError("--tile must be 8, 16, or 32");
}

void SetOption(Options& options, const std::string& flag,
               const std::string& value) {
  if (flag == "--input") {
    options.input = value;
  } else if (flag == "--output") {
    options.output = value;
  } else if (flag == "--mode") {
    if (value != "naive" && value != "tiled") {
      throw InputError("--mode must be naive or tiled");
    }
    options.tiled = value == "tiled";
  } else if (flag == "--tile") {
    options.tile = ParseTile(value);
  } else {
    throw InputError("Unknown option: " + flag);
  }
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; i += 2) {
    const std::string flag = argv[i];
    if (i + 1 == argc) {
      throw InputError("Missing value for " + flag);
    }
    const std::string value = argv[i + 1];
    if (value.empty() || value.compare(0, 2, "--") == 0) {
      throw InputError("Missing value for " + flag);
    }
    SetOption(options, flag, value);
  }
  if (options.input.empty() || options.output.empty()) {
    throw InputError("--input and --output are required");
  }
  return options;
}

int ReadDimension(std::istream& file) {
  std::string token;
  if (!(file >> token)) {
    throw InputError("Expected M K N");
  }
  const char* first = token.data();
  const char* last = first + token.size();
  if (*first == '+') {
    ++first;
  }
  int value = 0;
  const auto parsed = std::from_chars(first, last, value);
  if (parsed.ec != std::errc{} || parsed.ptr != last || value <= 0) {
    throw InputError("M, K and N must be positive integers");
  }
  return value;
}

std::size_t ElementCount(int rows, int columns) {
  const auto size = static_cast<std::size_t>(rows);
  if (size > std::vector<float>().max_size() / columns) {
    throw InputError("Matrix dimensions are too large");
  }
  return size * columns;
}

std::vector<float> ReadMatrix(std::istream& file, std::size_t count,
                              const char* name) {
  std::vector<float> matrix;
  std::string token;
  for (std::size_t i = 0; i < count; ++i) {
    if (!(file >> token)) {
      throw InputError(std::string("Missing value in ") + name);
    }
    char* end = nullptr;
    const float value = std::strtof(token.c_str(), &end);
    if (end != token.c_str() + token.size() || !std::isfinite(value)) {
      throw InputError(std::string("Invalid FP32 value in ") + name);
    }
    matrix.push_back(value);
  }
  return matrix;
}

Matrices ReadInput(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw InputError("Cannot open input: " + path);
  }
  const Shape shape{ReadDimension(file), ReadDimension(file),
                    ReadDimension(file)};
  ElementCount(shape.m, shape.n);
  auto a = ReadMatrix(file, ElementCount(shape.m, shape.k), "A");
  auto b = ReadMatrix(file, ElementCount(shape.k, shape.n), "B");
  std::string extra;
  if (file >> extra || file.bad()) {
    throw InputError("Expected exactly M*K + K*N matrix values");
  }
  return Matrices{shape, std::move(a), std::move(b)};
}

void WriteOutput(const std::string& path, Shape shape,
                 const std::vector<float>& result) {
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Cannot open output: " + path);
  }
  file << shape.m << ' ' << shape.n << '\n';
  file << std::setprecision(std::numeric_limits<float>::max_digits10);
  for (int i = 0; i < shape.m; ++i) {
    for (int j = 0; j < shape.n; ++j) {
      if (j > 0) {
        file << ' ';
      }
      file << result[static_cast<std::size_t>(i) * shape.n + j];
    }
    file << '\n';
  }
  file.close();
  if (!file) {
    throw std::runtime_error("Cannot write output: " + path);
  }
}

void Run(const Options& options) {
  const Matrices input = ReadInput(options.input);
  const Shape shape = input.shape;
  DeviceMatrices gpu;
  gpu.Upload(shape, input.a.data(), input.b.data());
  LaunchMatmul(gpu.a, gpu.b, gpu.c, shape, options.tiled, options.tile);
  std::vector<float> result(ElementCount(shape.m, shape.n));
  CheckCuda(cudaMemcpy(result.data(), gpu.c, result.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));
  for (const float value : result) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("The result is outside the finite FP32 range");
    }
  }
  WriteOutput(options.output, shape, result);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    PrintHelp();
    return 0;
  }
  try {
    Run(ParseOptions(argc, argv));
    return 0;
  } catch (const InputError& error) {
    std::cerr << error.what() << '\n';
    return 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
