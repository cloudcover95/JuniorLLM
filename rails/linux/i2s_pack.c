/* I2_S pack: 0->0, 1->1, -1->2. Two bits per trit. */
int i2s_pack(const int *t, int n, unsigned char *out) {
  unsigned acc = 0;
  int bits = 0, k = 0, i;
  for (i = 0; i < n; i++) {
    int c = t[i] == 0 ? 0 : (t[i] == 1 ? 1 : 2);
    acc = (acc << 2) | (unsigned)c;
    bits += 2;
    if (bits == 8) {
      out[k++] = (unsigned char)acc;
      acc = 0;
      bits = 0;
    }
  }
  if (bits) out[k++] = (unsigned char)(acc << (8 - bits));
  return k;
}
