#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 64

static inline int min(int a, int b) { return a < b ? a : b; }

int main(void) {
    int n;
    if (fread(&n, sizeof(int), 1, stdin) != 1) return 1;

    float *a = malloc(n * n * sizeof(float));
    float *b = malloc(n * n * sizeof(float));
    float *c = calloc(n * n, sizeof(float));
    if (!a || !b || !c) return 1;

    if (fread(a, sizeof(float), n * n, stdin) != (size_t)(n * n)) return 1;
    if (fread(b, sizeof(float), n * n, stdin) != (size_t)(n * n)) return 1;

    /* Tiled matrix multiply - blocks fit in L1 cache */
    for (int bi = 0; bi < n; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < n; bj += BLOCK_SIZE) {
            for (int bk = 0; bk < n; bk += BLOCK_SIZE) {
                int i_end = min(bi + BLOCK_SIZE, n);
                int j_end = min(bj + BLOCK_SIZE, n);
                int k_end = min(bk + BLOCK_SIZE, n);
                for (int i = bi; i < i_end; i++) {
                    for (int k = bk; k < k_end; k++) {
                        float a_ik = a[i * n + k];
                        for (int j = bj; j < j_end; j++) {
                            c[i * n + j] += a_ik * b[k * n + j];
                        }
                    }
                }
            }
        }
    }

    fwrite(c, sizeof(float), n * n, stdout);

    free(a);
    free(b);
    free(c);
    return 0;
}
