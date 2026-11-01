// 1-bit error-feedback sigma-delta: y = x + NTF*e, NTF = B/A (b0=a0=1). Reads f64 stdin, writes packed bits LSB-first.
#include <stdio.h>
#include <stdlib.h>
#define N 5
int main(int argc, char **argv) {
  double a[N+1], b[N+1], c[N+1], ew[N+1] = {0}, ww[N+1] = {0};
  for (int k = 0; k <= N; k++) { b[k] = atof(argv[1+k]); a[k] = atof(argv[2+N+k]); c[k] = b[k] - a[k]; }
  double x; unsigned char byte = 0; int nb = 0; long clip = 0;
  while (fread(&x, 8, 1, stdin) == 1) {
    double w = 0;
    for (int k = 1; k <= N; k++) w += c[k]*ew[k] - a[k]*ww[k];
    double v = x + w, y = v >= 0 ? 1 : -1, e = y - v;
    if (v > 4 || v < -4) clip++;
    for (int k = N; k > 1; k--) { ew[k] = ew[k-1]; ww[k] = ww[k-1]; }
    ew[1] = e; ww[1] = w;
    if (y > 0) byte |= 1 << nb;
    if (++nb == 8) { fputc(byte, stdout); byte = 0; nb = 0; }
  }
  fprintf(stderr, "large states: %ld\n", clip);
  return 0;
}
