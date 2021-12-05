#include "generated.h"
#include "matrix.h"
#include <random>
#include <ruy/ruy.h>
#include <utility>

// This file argmaxes for best ordering of A, B
// Only on ARM. No intgemm.

using namespace pg;

std::vector<std::tuple<Order, Order, Order>> Orderings() {
  std::vector<std::tuple<Order, Order, Order>> result;
  std::vector<Order> opt = {Order::RowMajor, Order::ColMajor};
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) {
      for (size_t k = 0; k < 2; k++) {
        result.emplace_back(opt[i], opt[j], opt[k]);
      }
    }
  }
  return result;
}

int main() {
  std::mt19937_64 gen64;
  gen64.seed(42);

  // Flag to quickly prototype and inspect without full data.
  bool prototyping = false;

  // How many times do we run multiply, so the cost of multiply dominates other
  // costs and ranking becomes clear?
  const size_t MONTE_CARLO_RUNS = 1000;
  auto orderings = Orderings();

  std::vector<double> durations;

  for (auto &dimensions : PROBLEM_SIZES) {
    for (auto &ordering : orderings) {
      auto [a_order, b_order, c_order] = ordering;
      auto [M, N, P] = unroll(dimensions);

      // Creation, deletion might take a bit of time. Might want to do a lot of
      // multiplies to make that factor standalone.
      auto [A, B, bias] = generateIntegralInput(gen64, M, N, P, ordering);

      using DestScalar = std::int32_t;
      using AccumScalar = std::int32_t;

      // Ensuring allocation troubles end here.
      // Actual storage for ruy skeleton, we control this
      Layout productLayout(A.layout().rows(), B.layout().cols(), c_order);
      Matrix<DestScalar> C(productLayout);

      // Begin measuring now.
      auto start = std::chrono::steady_clock::now();
      for (size_t i = 0; i < MONTE_CARLO_RUNS; i++) {

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
        ruy::Matrix<AccumScalar> dst;
        ruy::MakeSimpleLayout(C.layout().rows(), C.layout().cols(),
                              convertToRuy(C.layout().order()),
                              dst.mutable_layout());

        dst.set_data(C.data());

        // When Dst is int32, mul_params is unused.
        ruy::MulParams<AccumScalar, DestScalar> mul_params;
        ruy::Mul(lhs, rhs, mul_params, &context, &dst);
      }

      auto duration = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - start)
                          .count();

      durations.push_back(duration);

      if (prototyping) {
        break;
      }
    }
  }

  // Revisit durations vector in th the same order, print this time, outside the
  // benchmarking code.
  auto pDuration = durations.begin();
  for (auto &dimensions : PROBLEM_SIZES) {
    for (auto &ordering : orderings) {
      auto [a_order, b_order, c_order] = ordering;
      auto [M, N, P] = unroll(dimensions);
      auto toString = [](const Order &order) {
        return order == Order::RowMajor ? "RowMajor" : "ColMajor";
      };

      std::cout << M << "x" << N << "x" << P << " | ";
      std::cout << "(" << toString(a_order) << ", " << toString(b_order) << ", "
                << toString(c_order) << ") | ";

      if (pDuration == durations.end()) {
        break;
      }
      std::cout << *pDuration++ << std::endl;
    }
  }
  return 0;
}
