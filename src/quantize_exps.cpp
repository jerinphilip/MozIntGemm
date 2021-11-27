#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

using Index = std::int32_t;
#include <arm_neon.h>

void fastQuantize(const float *input, float scale, float zero_point, Index rows,
                  Index width, int8_t *output) {
  const float32x4_t *Input = reinterpret_cast<const float32x4_t *>(input);
  const float32x4_t *InputEnd =
      reinterpret_cast<const float32x4_t *>(input + rows * width);

  int8x8_t *Output = reinterpret_cast<int8x8_t *>(output);
  while (Input != InputEnd) {
    // Vector multiply by scalar
    // float32x4_t vmulq_n_f32(float32x4_t a, float32_t b);
    // VMUL.F32 q0,q0,d0[0]
    float32x4_t scaledFloat_lo = vmulq_n_f32(*Input++, scale);

    // Convert from float
    // int32x4_t  vcvtnq_s32_f32(float32x4_t a);
    // VCVT.S32.F32 q0, q0
    int32x4_t scaledInt_lo = vcvtnq_s32_f32(scaledFloat_lo);

    // Vector saturating narrow integer
    // int16x4_t  vqmovn_s32(int32x4_t a);   // VQMOVN.S32 d0,q0
    int16x4_t s16x4_lo = vqmovn_s32(scaledInt_lo);

    // Vector multiply by scalar
    // float32x4_t vmulq_n_f32(float32x4_t a, float32_t b);
    // VMUL.F32 q0,q0,d0[0]
    float32x4_t scaledFloat_hi = vmulq_n_f32(*Input++, scale);

    // Convert from float
    // int32x4_t  vcvtnq_s32_f32(float32x4_t a);
    // VCVT.S32.F32 q0, q0
    int32x4_t scaledInt_hi = vcvtnq_s32_f32(scaledFloat_hi);

    // Vector saturating narrow integer
    // int16x4_t  vqmovn_s32(int32x4_t a);
    // VQMOVN.S32 d0,q0
    int16x4_t s16x4_hi = vqmovn_s32(scaledInt_hi);

    // Combine two ints.
    // int16x8_t   vcombine_s16(int16x4_t low, int16x4_t high);
    int16x8_t s16x8 = vcombine_s16(s16x4_lo, s16x4_hi);

    // Vector saturating narrow integer
    int8x8_t s8x8 = vqmovn_s16(s16x8);

    *Output = s8x8;
    ++Output;
  };
}

void slowQuantize(const float *input, float scale, float zero_point, Index rows,
                  Index width, int8_t *output) {
  // Dumb quantize we will improve this eventually.
  const Index size = rows * width;
  for (size_t i = 0; i < size; i++) {
    float value = roundf(scale * input[i]);
    value = std::max(-127.0f, value);
    value = std::min(127.0f, value);
    output[i] = static_cast<int8_t>(value);
  };
}

#include <iostream>

int main() {
  std::mt19937_64 gen64;
  const size_t rows = 16, cols = 16;
  std::vector<float> A(rows * cols);
  std::vector<int8_t> output(rows * cols);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::generate(A.begin(), A.end(), [&]() { return dist(gen64); });

#define ChooseQuantize(fn)                                                     \
  std::cout << #fn << std::endl;                                               \
  fn(A.data(), 127.0f, 0, rows, cols, output.data());                          \
  for (size_t i = 0; i < rows * cols; i++) {                                   \
    std::cout << (double)output[i] << " ";                                     \
  }                                                                            \
  std::cout << std::endl;

  ChooseQuantize(fastQuantize);
  ChooseQuantize(slowQuantize);
  return 0;
}
