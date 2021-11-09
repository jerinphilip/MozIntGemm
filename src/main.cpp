#include "firefox_interface.h"
#include "matrix.h"
#include <iostream>
#include <random>

int main(int argc, char **argv) {
  // Need 8 bit ints, that's all.
  std::mt19937_64 gen64;
  gen64.seed(42);

  constexpr size_t DIM_MAX = 4;
  constexpr size_t DIM_MIN = 2;
  std::uniform_int_distribution<> distribution(DIM_MIN, DIM_MAX);

  size_t M, N, P; // A = M x N matrix, B = N x P

  M = distribution(gen64);
  N = distribution(gen64);
  P = distribution(gen64);

  // Do some stuff to get stuff rounded to multiples of 8
  // const size_t _WIDTH = 32;
  // M = ((M / _WIDTH) + 1) * _WIDTH;
  // N = ((N / _WIDTH) + 1) * _WIDTH;
  // P = ((P / _WIDTH) + 1) * _WIDTH;
  //
  M = 1, N = 16, P = 8;

  using pg::Matrix;

  std::cout << M << " " << N << " " << P << "\n\n";
  Matrix<int8_t> A(M, N), mB_prepared(N, P);
  Matrix<float> B(N, P);

  A.fill(gen64);
  B.fill(gen64);

  Matrix<float> bias(1, P);
  // bias.fill(gen64);

  std::cout << "A:\n" << A << "\n";
  std::cout << "B:\n" << B << "\n";

  std::cout << "bias:\n" << bias << "\n";

  Matrix<float> ruyProduct(M, P), intgemmProduct(M, P);

  const int8_t *A_prepared = A.data();
  int8_t *B_prepared = mB_prepared.data();
  const float *bias_prepared = bias.data();
  float *output;

  output = intgemmProduct.data();
  pg::Intgemm::int8PrepareB(B.data(), /*scale=*/1.0f, /*zero_point=*/0.0f,
                            /*width=*/N, /*cols_B=*/P, B_prepared);
  pg::Intgemm::int8MultiplyAndAddBias(
      A_prepared, /*scale=*/1.0, /*zero_point=*/0.0f, B_prepared, /*scale=*/1.0,
      /*zero_point=*/0.0f, bias_prepared, /*scale_output=*/1.0f, M, N, P,
      output);

  std::cout << "Intgemm A*B : \n" << intgemmProduct << "\n";

  Matrix<int8_t> BForRuy(N, P);
  BForRuy.fill(B);
  output = ruyProduct.data();
  pg::Ruy::int8MultiplyAndAddBias(A_prepared, /*scale=*/1.0,
                                  /*zero_point=*/0.0f, BForRuy.data(),
                                  /*scale=*/1.0,
                                  /*zero_point=*/0.0f, bias_prepared,
                                  /*scale_output=*/1.0f, M, N, P, output);

  std::cout << "Ruy A*B : \n" << ruyProduct << "\n";

  std::cout << "Mean-Squared-Error(ruyProduct, intgemmProduct) = "
            << pg::MeanSquaredError(ruyProduct, intgemmProduct) << "\n";
  return 0;
}
