/* Bit-plane matvec. Pos plane minus neg plane. No LUT. */
#include <stddef.h>
#include <stdint.h>

int junior_bitplane(const int8_t *T, const float *x, float *y, float gamma, size_t m, size_t k) {
  size_t i, j;
  if (!T || !x || !y || m == 0 || k == 0) return -1;
  for (i = 0; i < m; i++) {
    float pos = 0.f, neg = 0.f;
    const int8_t *row = T + i * k;
    for (j = 0; j < k; j++) {
      int8_t t = row[j];
      if (t > 0) pos += x[j];
      else if (t < 0) neg += x[j];
    }
    y[i] = gamma * (pos - neg);
  }
  return 0;
}
