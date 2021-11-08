#include "firefox_interface.h"
#include "ruy/ruy.h"
#include <memory>

namespace pg::Ruy {

void quantize(const float *input, float scale, float zero_point, Index rows,
              Index width, int8_t *output) {
  // Dumb quantize we will improve this eventually.
  const Index size = rows * width;
  for (size_t i = 0; i < size; i++) {
    output[i] = static_cast<int8_t>(input[i] * scale - zero_point);
  };
}

// Unquantizes int32_t accumulated output from int8_t multiplies, corrects for
// any shifts and writes result to output.
void unquantize(const int32_t *input, float scale, float zero_point, Index rows,
                Index cols, float *output) {

  const Index size = rows * cols;
  float unquant_multiplier = 1.0f / (scale * scale);
  for (size_t i = 0; i < size; i++) {
    output[i] = static_cast<float>(input[i] * unquant_multiplier) + zero_point;
  }
}
void int8PrepareB(const float *input_B, float scale, float zero_point,
                  Index width, Index cols_B, int8_t *output) {
  quantize(input_B, scale, zero_point, width, cols_B, output);
}

void int8PrepareBFromTransposed(const float *input_B_transposed, float scale,
                                float zero_point, Index width, Index cols_B,
                                int8_t *output) {}

void int8PrepareBFromQuantizedTransposed(const int8_t *input_B_quant_transposed,
                                         Index width, Index cols_B,
                                         int8_t *output) {}

void int8PrepareA(const float *input_A, float scale, float zero_point,
                  Index rows_A, Index width, int8_t *output) {
  quantize(input_A, scale, zero_point, rows_A, width, output);
}

void int8PrepareBias(const int8_t *input_B_prepared, float scale,
                     float zero_point, Index width, Index cols_B,
                     const float *input_bias, float *output) {}

void int8MultiplyAndAddBias(const int8_t *input_A_prepared, float scale_A,
                            float zero_point_A, const int8_t *input_B_prepared,
                            float scale_B, float zero_point_B,
                            const float *input_bias_prepared, Index rows_A,
                            Index width, Index cols_B, float *output) {

  // It is expected that somehow we have managed to call all prepare by the time
  // we are here, with inputs (prepared) in int8_t. All that's left to do is use
  // ruy for multiply and then start with the reverse ops to get to fp32.

  // Use ruy to multiply.
  // The following is adapted from
  // https://github.com/google/ruy/blob/878283640de7946a43053e8ebf4f15114fbc9156/example/example.cc#L129-L152
  ruy::Context context;
  ruy::Matrix<std::int8_t> lhs;
  ruy::MakeSimpleLayout(rows_A, width, ruy::Order::kRowMajor,
                        lhs.mutable_layout());
  lhs.set_data(input_A_prepared);
  ruy::Matrix<std::int8_t> rhs;
  ruy::MakeSimpleLayout(width, cols_B, ruy::Order::kRowMajor,
                        rhs.mutable_layout());
  rhs.set_data(input_B_prepared);
  ruy::Matrix<std::int32_t> dst;
  ruy::MakeSimpleLayout(rows_A, cols_B, ruy::Order::kRowMajor,
                        dst.mutable_layout());

  std::unique_ptr<std::int32_t> dst_data =
      std::make_unique<std::int32_t>(rows_A * cols_B);
  dst.set_data(dst_data.get());

  // When Dst is int32, mul_params is unused.
  ruy::MulParams<std::int32_t, std::int32_t> mul_params;
  ruy::Mul(lhs, rhs, mul_params, &context, &dst);

  // Convert to float (this is done through unquantize for now)
  unquantize(dst_data.get(), scale_A, zero_point_A, rows_A, cols_B, output);

  // TODO(jerinphilip) There's some bias writing left.
  //
}

void int8SelectColumnsOfB(const int8_t *input_B_prepared, Index width,
                          Index cols_B, const Index *cols, const Index num_cols,
                          int8_t *output) {}

} // namespace pg::Ruy
