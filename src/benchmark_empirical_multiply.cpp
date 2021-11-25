#include "generated.h"
#include "matrix.h"
#include <random>
#include <ruy/ruy.h>
#include <utility>


// This file argmaxes for best ordering of A, B
// Only on ARM. No intgemm. 

using namespace pg;

std::vector<std::pair<Order, Order>> ABOrderings(){
    std::vector<std::pair<Order, Order>> result;
    for(size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 2; j++){
            Order a_order = (i == 0)? Order::RowMajor: Order::ColMajor;
            Order b_order = (j == 0)? Order::RowMajor: Order::ColMajor;
            result.emplace_back(a_order, b_order);
        }
    }
    return result;
}

int main(){
    std::mt19937_64 gen64;
    gen64.seed(42);

    const size_t MONTE_CARLO_RUNS = 1000;

    auto orderings = ABOrderings();
    for(auto &ordering: orderings){
        // I need reflection, C++23, where art thou?
        std::cout << (size_t)ordering.first << " " << (size_t)ordering.second << std::endl;
    }

    for(auto &ordering: orderings){
        for(auto &dimensions: PROBLEM_SIZES){
            auto [M, N, P] = unroll(dimensions);
            std::cout << M << " " << N << " " << P << std::endl;
            // Creation, deletion might take a bit of time. Might want to do a lot of multiplies to make that factor standalone.
            // TODO: Parameterize this by ordering.
            auto [A, B, bias] = generateIntegralInput(gen64, M, N, P);

            for(size_t i = 0; i < MONTE_CARLO_RUNS; i++){
                // We're only doing A*B if we look at it.
                using DestScalar = std::int32_t;
                using AccumScalar = std::int32_t;

                ruy::Context context;

                // A matrix.
                ruy::Matrix<int8_t> lhs;
                ruy::MakeSimpleLayout(A.layout().rows(), A.layout().cols(), ruy::Order::kRowMajor,
                        lhs.mutable_layout());
                lhs.set_data(A.data());


                // B matrix.
                ruy::Matrix<int8_t> rhs;
                ruy::MakeSimpleLayout(B.layout().rows(), B.layout().cols(), ruy::Order::kColMajor,
                        rhs.mutable_layout());
                rhs.set_data(B.data());

                // Setup product, in ruy skeleton.
                Layout productLayout(A.layout().rows(), B.layout().cols(), Order::ColMajor);
                ruy::Matrix<AccumScalar> dst;
                ruy::MakeSimpleLayout(productLayout.rows(), productLayout.cols(), ruy::Order::kRowMajor,
                        dst.mutable_layout());

                // Actual storage for ruy skeleton, we control this
                Matrix<AccumScalar> dst_data(productLayout);
                AccumScalar *dest_ptr = dst_data.data();
                dst.set_data(dest_ptr);

                // When Dst is int32, mul_params is unused.
                ruy::MulParams<AccumScalar, DestScalar> mul_params;
                ruy::Mul(lhs, rhs, mul_params, &context, &dst);
            }
        }

    }
    return 0;
}

