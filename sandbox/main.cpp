#include "ruy/ruy.h"
#include <iostream>

int main(int argc, char **argv) {
  ruy::Context context;
  const std::int8_t lhs_data[] = {1, 2, 3, 4};
  const std::int8_t rhs_data[] = {1, 2, 3, 4};
  std::int32_t dst_data[4];

  ruy::Matrix<std::int8_t> lhs;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kRowMajor, lhs.mutable_layout());
  lhs.set_data(lhs_data);
  ruy::Matrix<std::int8_t> rhs;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kColMajor, rhs.mutable_layout());
  rhs.set_data(rhs_data);
  ruy::Matrix<std::int32_t> dst;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kColMajor, dst.mutable_layout());
  dst.set_data(dst_data);

  // When Dst is int32, mul_params is unused.
  ruy::MulParams<std::int32_t, std::int32_t> mul_params;
  ruy::Mul(lhs, rhs, mul_params, &context, &dst);

  std::cout << "Example Mul, returning raw int32 accumulators:\n";
  std::cout << "LHS:\n" << lhs;
  std::cout << "RHS:\n" << rhs;
  std::cout << "Result:\n" << dst << "\n";
  return 0;
}
