// @jerinphilip is sorry about this abomination. This should never have had to
// exist.

#include "3rd-party/intgemm/intgemm/callbacks/configs.h"
#include "impl_ruy-export.cpp"
#include <cstdint>

namespace intgemm {

using Index = std::uint32_t;

struct TileInfo {
  const Index a_rows;
  const Index a_cols;
  const Index b_rows;
  const Index b_cols;
};

template <class Integer> struct IntTmpl {
  using Integer = Integer;

  // A's size must be a multiple of 1x64, B's size must be a multiple of 64x8.
  static constexpr TileInfo tile_info{1, 64, 64, 8};

  // Currently A is prepared by quantization but this could theoretically
  // change. A's columns must be a multiple of 8. The number of rows is
  // anything.
  static inline void PrepareA(const float *input, Integer *output,
                              float quant_mult, Index rows, Index cols) {
    int8PrepareA(input, /*scale=*/quant_mult, /*zero_point=*/0, rows, cols,
                 output);
  }

  // Warning: the output of PrepareB depends on the CPU.
  // It will match the Multiply function on the same CPU though.
  static void PrepareB(const float *input, Integer *output, float quant_mult,
                       Index rows, Index cols) {
    int8PrepareB(input, /*scale=*/quant_mult, /*zero_point=*/0, rows, cols,
                 output);
  }

  // Convert from a B that was already transposed (routine not provided) and
  // quantized (e.g. with Quantize) to the CPU-dependent format used for
  // Multiply.  This is useful for storing a quantized model on disk then in a
  // CPU-independent fashion.
  static void PrepareBQuantizedTransposed(const Integer *input, Integer *output,
                                          Index inner,
                                          Index B_untransposed_cols) {
    int8PrepareBFromQuantizedTransposed(input, inner, B_untransposed_cols,
                                        output);
  }

  // Convert from a B that was already transposed (routine not provided) to
  // the CPU-dependent format used for Multiply.  This is useful for storing
  // a quantized model on disk then in a CPU-independent fashion.
  static void PrepareBTransposed(const float *input, Integer *output,
                                 float quant_mul, Index inner,
                                 Index B_untransposed_cols) {
    int8PrepareBFromTransposed(input, /*scale=*/quant_mult, /*zero_point=*/0,
                               inner, B_untransposed_cols, output)
  }

  // Select columns from a prepared B matrix.  The number of selected columns
  // must be a multiple of 8.
  static void SelectColumnsB(const Integer *input, Integer *output, Index rows,
                             const Index *cols_begin, const Index *cols_end) {
    int8SelectColumnsOfB(input, rows, cols_begin,
                         std::distance(cols_begin, cols_end), output);
  }

  // Multiply C = A * B, presuming A and B have been prepared.
  template <class Callback>
  static void Multiply(const Integer *A, const Integer *B, Index A_rows,
                       Index width, Index B_cols, Callback callback) {

    /*
    void int8MultiplyAndAddBias(
        const int8_t *input_A_prepared, float scale_A, float zero_point_A,
        const int8_t *input_B_prepared, float scale_B, float zero_point_B,
        const float *input_bias_prepared, float unquant_multiplier,
        Index rows_A, Index width, Index cols_B, float *output);
    */

    // clang-format off
    int8MultiplyAndAddBias(
        A, /*scale_A=*/1.0f, /*zero_point_A=*/0, 
        B, /*scale_B=*/0.0f, /*zero_point_B=*/0, 
        callback.bias_prepared, callback.unquant_mult,
        A_rows, width, B_cols, callback.output_addr);
    // clang-format on

    /*
    intgemm::Int8Shift::Multiply(
        input_A_prepared, input_B_prepared, rows_A, width, cols_B,
        intgemm::callbacks::UnquantizeAndAddBiasAndWrite(
            unquant_factor, input_bias_prepared, output));
    */
  }

  static const char *const kName = "8-bit Ruy masquerading as intgemm";
};

using Int8 = IntTmpl<int8_t>;
using Int16 = IntTmpl<int16_t>;

} // namespace intgemm
