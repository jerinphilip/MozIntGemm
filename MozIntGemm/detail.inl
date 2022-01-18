namespace detail {

template <class T> class AlignedVector {
public:
  AlignedVector(size_t num_elem)
      : size_(num_elem),
        storage_(reinterpret_cast<T *>(
            ruy::detail::SystemAlignedAlloc(sizeof(T) * num_elem))) {}

  T *begin() { return storage_; }
  T *data() { return storage_; }
  size_t size() const { return size_; }
  size_t memSize() const { return sizeof(T) * size_; }

  // Forbid copy
  AlignedVector(const AlignedVector &) = delete;
  AlignedVector &operator=(const AlignedVector &) = delete;

  ~AlignedVector() {
    ruy::detail::SystemAlignedFree(reinterpret_cast<void *>(storage_));
  }

private:
  T *storage_;
  size_t size_;
};

// TODO: Workout similar to ruy. enum value is causing type/value complaints.
// Intentions are to get the functions to exist together without flouting
// One-Definition-Rule (ODR).
struct kStandardCpp {};
struct kNeon {};

#if RUY_PLATFORM_NEON
using kHighestPath = kNeon;
#else
using kHighestPath = kStandardCpp;
#endif

template <class Path> struct Preprocess {
  static void quantize(const float *input, float scale, float zero_point,
                       Index rows, Index width, int8_t *output) {
    const Index size = rows * width;
    for (size_t i = 0; i < size; i++) {
      // Round to nearest after multiplying with scale.
      float value = roundf(scale * input[i]);

      // Since float can store bigger values, we threshold anything that's gone
      // higher and can't fit in int8.
      value = std::max<float>(-127.0f, value);
      value = std::min<float>(127.0f, value);

      // Finally a static cast.
      output[i] = static_cast<int8_t>(value);
    };
  }

  template <class Scalar>
  static void transpose(const Scalar *input, Index rows, Index cols,
                        Scalar *output) {
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        output[j * rows + i] = input[i * cols + j];
      }
    }
  }
  static void unquantizeAddBias(const int32_t *input,
                                const float *input_bias_prepared,
                                float unquant_multiplier, Index rows_A,
                                Index cols_B, float *output) {
    for (size_t i = 0; i < rows_A; i++) {
      for (size_t j = 0; j < cols_B; j++) {
        Index idx = i * cols_B + j;
        output[idx] =
            (input[idx] * unquant_multiplier) + input_bias_prepared[j];
      }
    }
  }
};

#if RUY_PLATFORM_NEON
template <> struct Preprocess<kNeon> {
  static void quantize(const float *input, float scale, float zero_point,
                       Index rows, Index width, int8_t *output) {
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

  template <class Scalar>
  static void transpose(const Scalar *input, Index rows, Index cols,
                        Scalar *output) {
    // We're fixing this.. are we?
    return;
  }

  static void transpose_16x16(const int8_t *src, int8_t *dst) {
    constexpr size_t width = 16;
    // clang-format off
    int8x16x2_t r0 = vtrnq_s8(vld1q_s8(&src[ 0*width]), vld1q_s8(&src[ 1*width]));
    int8x16x2_t r1 = vtrnq_s8(vld1q_s8(&src[ 2*width]), vld1q_s8(&src[ 3*width]));
    int8x16x2_t r2 = vtrnq_s8(vld1q_s8(&src[ 4*width]), vld1q_s8(&src[ 5*width]));
    int8x16x2_t r3 = vtrnq_s8(vld1q_s8(&src[ 6*width]), vld1q_s8(&src[ 7*width]));
    int8x16x2_t r4 = vtrnq_s8(vld1q_s8(&src[ 8*width]), vld1q_s8(&src[ 9*width]));
    int8x16x2_t r5 = vtrnq_s8(vld1q_s8(&src[10*width]), vld1q_s8(&src[11*width]));
    int8x16x2_t r6 = vtrnq_s8(vld1q_s8(&src[12*width]), vld1q_s8(&src[13*width]));
    int8x16x2_t r7 = vtrnq_s8(vld1q_s8(&src[14*width]), vld1q_s8(&src[15*width]));


    int16x8x2_t t0 = vtrnq_s16(vreinterpretq_s16_s8(r0.val[0]), vreinterpretq_s16_s8(r1.val[0]));
    int16x8x2_t t1 = vtrnq_s16(vreinterpretq_s16_s8(r2.val[0]), vreinterpretq_s16_s8(r3.val[0]));
    int16x8x2_t t2 = vtrnq_s16(vreinterpretq_s16_s8(r4.val[0]), vreinterpretq_s16_s8(r5.val[0]));
    int16x8x2_t t3 = vtrnq_s16(vreinterpretq_s16_s8(r6.val[0]), vreinterpretq_s16_s8(r7.val[0]));
    int16x8x2_t t4 = vtrnq_s16(vreinterpretq_s16_s8(r0.val[1]), vreinterpretq_s16_s8(r1.val[1]));
    int16x8x2_t t5 = vtrnq_s16(vreinterpretq_s16_s8(r2.val[1]), vreinterpretq_s16_s8(r3.val[1]));
    int16x8x2_t t6 = vtrnq_s16(vreinterpretq_s16_s8(r4.val[1]), vreinterpretq_s16_s8(r5.val[1]));
    int16x8x2_t t7 = vtrnq_s16(vreinterpretq_s16_s8(r6.val[1]), vreinterpretq_s16_s8(r7.val[1]));
    

    int32x4x2_t x0 = vtrnq_s32(vreinterpretq_s32_s16(t0.val[0]), vreinterpretq_s32_s16(t1.val[0]));
    int32x4x2_t x1 = vtrnq_s32(vreinterpretq_s32_s16(t4.val[0]), vreinterpretq_s32_s16(t5.val[0]));
    int32x4x2_t x2 = vtrnq_s32(vreinterpretq_s32_s16(t0.val[1]), vreinterpretq_s32_s16(t1.val[1]));
    int32x4x2_t x3 = vtrnq_s32(vreinterpretq_s32_s16(t4.val[1]), vreinterpretq_s32_s16(t5.val[1]));

    int32x4x2_t x4 = vtrnq_s32(vreinterpretq_s32_s16(t2.val[0]), vreinterpretq_s32_s16(t3.val[0]));
    int32x4x2_t x5 = vtrnq_s32(vreinterpretq_s32_s16(t6.val[0]), vreinterpretq_s32_s16(t7.val[0]));
    int32x4x2_t x6 = vtrnq_s32(vreinterpretq_s32_s16(t2.val[1]), vreinterpretq_s32_s16(t3.val[1]));
    int32x4x2_t x7 = vtrnq_s32(vreinterpretq_s32_s16(t6.val[1]), vreinterpretq_s32_s16(t7.val[1]));

    vst1q_s8(&dst[0*width], vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(x0.val[0]), vget_low_s32(x5.val[0])))); 
    vst1q_s8(&dst[1*width], vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(x1.val[0]), vget_low_s32(x6.val[0]))));
    vst1q_s8(&dst[2*width], vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(x2.val[0]), vget_low_s32(x4.val[0]))));
    vst1q_s8(&dst[3*width], vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(x3.val[0]), vget_low_s32(x6.val[1]))));
    vst1q_s8(&dst[4*width], vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(x0.val[1]), vget_low_s32(x4.val[1]))));
    vst1q_s8(&dst[5*width], vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(x1.val[1]), vget_low_s32(x7.val[0]))));
    vst1q_s8(&dst[6*width], vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(x2.val[1]), vget_low_s32(x5.val[1]))));
    vst1q_s8(&dst[7*width], vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(x3.val[1]), vget_low_s32(x7.val[1]))));
    vst1q_s8(&dst[8*width], vreinterpretq_s8_s32 (vcombine_s32(vget_high_s32(x0.val[0]), vget_high_s32(x5.val[0]))));
    vst1q_s8(&dst[9*width], vreinterpretq_s8_s32 (vcombine_s32(vget_high_s32(x1.val[0]), vget_high_s32(x6.val[0]))));
    vst1q_s8(&dst[10*width], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x2.val[0]), vget_high_s32(x4.val[0]))));
    vst1q_s8(&dst[11*width], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x3.val[0]), vget_high_s32(x6.val[1]))));
    vst1q_s8(&dst[12*width], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x0.val[1]), vget_high_s32(x4.val[1]))));
    vst1q_s8(&dst[13*width], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x1.val[1]), vget_high_s32(x7.val[0]))));
    vst1q_s8(&dst[14*width], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x2.val[1]), vget_high_s32(x5.val[1]))));
    vst1q_s8(&dst[15*width], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x3.val[1]), vget_high_s32(x7.val[1]))));

    // clang-format on
  }

  // Specialization for int8_t
  static void transpose(const int8_t *input, Index rows, Index cols,
                        int8_t *output) {}

  static void unquantizeAddBias(const int32_t *input,
                                const float *input_bias_prepared,
                                float unquant_multiplier, Index rows_A,
                                Index cols_B, float *output) {
    // Set all registers in lane from same scalar value.
    float32x4_t multiplier = vdupq_n_f32(unquant_multiplier);
    const int32x4_t *Input = reinterpret_cast<const int32x4_t *>(input);
    const int32x4_t *InputEnd =
        reinterpret_cast<const int32x4_t *>(input + rows_A * cols_B);
    float32x4_t *Output = reinterpret_cast<float32x4_t *>(output);

    while (Input != InputEnd) {
      // Bias cycles every column for addition.
      const float32x4_t *Bias =
          reinterpret_cast<const float32x4_t *>(input_bias_prepared);

      // InputEnd needs to be determined to end the while loop below.
      const int32x4_t *RowEnd = reinterpret_cast<const int32x4_t *>(
          reinterpret_cast<const int32_t *>(Input) + cols_B);

      while (Input != RowEnd) {
        // Operation happening for 4-elements together:
        // output = [int32_t]input * [float]quant_mult + [float]bias;
        float32x4_t floatInput = vcvtq_f32_s32(*Input++);
        float32x4_t unquantized = vmulq_f32(floatInput, multiplier);
        *Output++ = vaddq_f32(unquantized, *Bias++);
      }
    }
  }
};
#endif
} // namespace detail
