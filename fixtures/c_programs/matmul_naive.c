#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int n;
    if (fread(&n, sizeof(int), 1, stdin) != 1) return 1;

    float *a = malloc(n * n * sizeof(float));
    float *b = malloc(n * n * sizeof(float));
    float *c = calloc(n * n, sizeof(float));
    if (!a || !b || !c) return 1;

    if (fread(a, sizeof(float), n * n, stdin) != (size_t)(n * n)) return 1;
    if (fread(b, sizeof(float), n * n, stdin) != (size_t)(n * n)) return 1;

    /* Naive i,j,k loop - B access is cache-hostile (column stride) */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }

    fwrite(c, sizeof(float), n * n, stdout);

    free(a);
    free(b);
    free(c);
    return 0;
}
