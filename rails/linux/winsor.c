/* Winsor p95. Same idea as junior_bitnet.winsor. Scalar. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static int cmpf(const void *a, const void *b) {
  float x = *(const float *)a, y = *(const float *)b;
  return (x > y) - (x < y);
}

int junior_winsor(const float *x, int8_t *t, float *gamma, size_t n) {
  size_t i, idx;
  float *absv, tau, g, acc = 0.f;
  if (!x || !t || !gamma || n == 0) return -1;
  absv = (float *)malloc(n * sizeof(float));
  if (!absv) return -2;
  for (i = 0; i < n; i++) absv[i] = fabsf(x[i]);
  qsort(absv, n, sizeof(float), cmpf);
  idx = (size_t)((95.0 / 100.0) * (n - 1));
  if (idx >= n) idx = n - 1;
  tau = absv[idx];
  for (i = 0; i < n; i++) acc += (absv[i] < tau) ? absv[i] : tau;
  free(absv);
  g = acc / (float)n + 1e-7f;
  *gamma = g;
  for (i = 0; i < n; i++) {
    float q = roundf(x[i] / g);
    if (q > 1.f) q = 1.f;
    if (q < -1.f) q = -1.f;
    t[i] = (int8_t)q;
  }
  return 0;
}
