/* Same math as junior_bitnet.math.absmean. Scalar. No unused NEON. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

int junior_absmean(const float *x, int8_t *t, float *gamma, size_t n) {
  size_t i;
  double acc = 0.0;
  float g;
  if (!x || !t || !gamma || n == 0) return -1;
  for (i = 0; i < n; i++) acc += fabsf(x[i]);
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
