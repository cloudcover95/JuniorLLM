/* Reduction SIMD. Quant scalar. No Rust. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

int junior_absmean(const float *x, int8_t *t, float *gamma, size_t n) {
  size_t i = 0;
  double acc = 0.0;
  float g;
  if (!x || !t || !gamma || n == 0) return -1;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  {
    float32x4_t vacc = vdupq_n_f32(0.f);
    for (; i + 4 <= n; i += 4) {
      float32x4_t v = vld1q_f32(x + i);
      vacc = vaddq_f32(vacc, vabsq_f32(v));
    }
    float tmp[4];
    vst1q_f32(tmp, vacc);
    acc = tmp[0] + tmp[1] + tmp[2] + tmp[3];
  }
#elif defined(__SSE2__)
  {
    __m128 vacc = _mm_setzero_ps();
    const __m128 sign = _mm_set1_ps(-0.f);
    for (; i + 4 <= n; i += 4) {
      __m128 v = _mm_loadu_ps(x + i);
      vacc = _mm_add_ps(vacc, _mm_andnot_ps(sign, v));
    }
    float tmp[4];
    _mm_storeu_ps(tmp, vacc);
    acc = tmp[0] + tmp[1] + tmp[2] + tmp[3];
  }
#endif
  for (; i < n; i++) acc += fabsf(x[i]);
  g = (float)(acc / (double)n) + 1e-7f;
  *gamma = g;
  for (i = 0; i < n; i++) {
    float q = roundf(x[i] / g);
    if (q > 1.f) q = 1.f;
    if (q < -1.f) q = -1.f;
    t[i] = (int8_t)q;
  }
  return 0;
}
