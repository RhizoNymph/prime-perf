#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    int w, h, iters;
    if (fread(&w, sizeof(int), 1, stdin) != 1) return 1;
    if (fread(&h, sizeof(int), 1, stdin) != 1) return 1;
    if (fread(&iters, sizeof(int), 1, stdin) != 1) return 1;

    size_t n = (size_t)w * h;
    float *old = calloc(n, sizeof(float));
    float *new = calloc(n, sizeof(float));
    if (!old || !new) return 1;

    if (fread(old, sizeof(float), n, stdin) != n) return 1;

    /* Copy boundary cells once — they never change */
    memcpy(new, old, n * sizeof(float));

    for (int iter = 0; iter < iters; iter++) {
        for (int i = 1; i < h - 1; i++) {
            for (int j = 1; j < w - 1; j++) {
                new[i * w + j] = (old[(i - 1) * w + j] +
                                  old[(i + 1) * w + j] +
                                  old[i * w + (j - 1)] +
                                  old[i * w + (j + 1)] +
                                  old[i * w + j]) / 5.0f;
            }
        }
        /* Swap old and new */
        float *tmp = old;
        old = new;
        new = tmp;
        /* Copy boundary cells into the new buffer for next iteration */
        if (iter < iters - 1) {
            /* Top row */
            memcpy(new, old, w * sizeof(float));
            /* Bottom row */
            memcpy(new + (h - 1) * w, old + (h - 1) * w, w * sizeof(float));
            /* Left and right columns */
            for (int i = 1; i < h - 1; i++) {
                new[i * w] = old[i * w];
                new[i * w + w - 1] = old[i * w + w - 1];
            }
        }
    }

    fwrite(old, sizeof(float), n, stdout);

    free(old);
    free(new);
    return 0;
}
