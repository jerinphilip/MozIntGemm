#include "detail.h"
#include "matrix.h"
#include <algorithm>
#include <iostream>
#include <random>

using namespace pg::Ruy::detail;
using namespace pg;

int main() {
  std::mt19937_64 gen64;
  const size_t M = 8, N = 64, P = 64;
  auto [A, B, bias] = generateInput(gen64, M, N, P);
  Matrix<int8_t> quantizedAStd(A.layout()), quantizedANeon(A.layout());

#define ChooseQuantize(ThePath, M, output)                                     \
  std::cout << #ThePath << std::endl;                                          \
  Preprocess<ThePath>::quantize(M.data(), 127.0f, 0, M.nrows(), M.ncols(),     \
                                output.data());                                \
  std::cout << output;

  ChooseQuantize(kStandardCpp, A, quantizedAStd);
  ChooseQuantize(kNeon, A, quantizedANeon);
  return 0;
}
