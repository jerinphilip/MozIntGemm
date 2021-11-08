#include "ffox_intgemm.h"
#include <iostream>
#include <random>

int main(int argc, char **argv) {
  // Need 8 bit ints, that's all.
  std::mt19937_64 gen64;
  gen64.seed(42);

  size_t M, N, P; // A = M x N matrix, B = N x P

  return 0;
}
