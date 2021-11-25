#include "generated.h"
#include "matrix.h"
#include <random>
#include <ruy/ruy.h>
#include <utility>

// This file argmaxes for best ordering of A, B
// Only on ARM. No intgemm.

using namespace pg;

std::vector<std::tuple<Order, Order, Order>> ABOrderings() {
  std::vector<std::tuple<Order, Order, Order>> result;
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) {
      for (size_t k = 0; k < 2; k++) {
        Order a_order = (i == 0) ? Order::RowMajor : Order::ColMajor;
        Order b_order = (j == 0) ? Order::RowMajor : Order::ColMajor;
        Order c_order = (k == 0) ? Order::RowMajor : Order::ColMajor;
        result.emplace_back(a_order, b_order, c_order);
      }
    }
  }
  return result;
}

int main() {
  std::mt19937_64 gen64;
  gen64.seed(42);

  const size_t MONTE_CARLO_RUNS = 1000;

  auto orderings = ABOrderings();
  for (auto &ordering : orderings) {
    // I need reflection, C++23, where art thou?
    auto [a, b, c] = ordering;
    // std::cout << size_t(a) << " " << size_t(b) << " " << size_t(c) <<
    // std::endl;
  }

  bool testing = false;
  for (auto &dimensions : PROBLEM_SIZES) {
    for (auto &ordering : orderings) {
      auto [a_order, b_order, c_order] = ordering;
      auto [M, N, P] = unroll(dimensions);
      // std::cout << M << " " << N << " " << P << std::endl;
      // Creation, deletion might take a bit of time. Might want to do a lot of
      // multiplies to make that factor standalone.
      // TODO: Parameterize this by ordering.
      auto [A, B, bias] = generateIntegralInput(gen64, M, N, P, ordering);

      auto start = std::chrono::steady_clock::now();
      for (size_t i = 0; i < MONTE_CARLO_RUNS; i++) {
        // We're only doing A*B if we look at it.
        using DestScalar = std::int32_t;
        using AccumScalar = std::int32_t;

        ruy::Context context;

        auto convertToRuy = [](const Order &order) {
          if (order == Order::RowMajor) {
            return ruy::Order::kRowMajor;
          } else {
            return ruy::Order::kColMajor;
          }
        };

        // A matrix.
        ruy::Matrix<int8_t> lhs;
        ruy::MakeSimpleLayout(A.layout().rows(), A.layout().cols(),
                              convertToRuy(A.layout().order()),
                              lhs.mutable_layout());
        lhs.set_data(A.data());

        // B matrix.
        ruy::Matrix<int8_t> rhs;
        ruy::MakeSimpleLayout(B.layout().rows(), B.layout().cols(),
                              convertToRuy(A.layout().order()),
                              rhs.mutable_layout());
        rhs.set_data(B.data());

        // Setup product, in ruy skeleton.
        Layout productLayout(A.layout().rows(), B.layout().cols(), c_order);

        ruy::Matrix<AccumScalar> dst;
        ruy::MakeSimpleLayout(productLayout.rows(), productLayout.cols(),
                              convertToRuy(productLayout.order()),
                              dst.mutable_layout());

        // Actual storage for ruy skeleton, we control this
        Matrix<AccumScalar> dst_data(productLayout);
        AccumScalar *dest_ptr = dst_data.data();
        dst.set_data(dest_ptr);

        // When Dst is int32, mul_params is unused.
        ruy::MulParams<AccumScalar, DestScalar> mul_params;
        ruy::Mul(lhs, rhs, mul_params, &context, &dst);
      }
      auto toString = [](const Order &order) {
        return order == Order::RowMajor ? "RowMajor" : "ColMajor";
      };

      std::cout << M << "x" << N << "x" << P << ": ";
      std::cout << "[A = " << toString(a_order) << ", B = " << toString(b_order)
                << " C = " << toString(c_order) << "] time: "
                << std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start)
                       .count()
                << "\n";
      if (testing) {
        break;
      }
    }
  }
  return 0;
}
