#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int compare(float a, float b) {
    int a_nan = isnan(a);
    int b_nan = isnan(b);
    if (a_nan && b_nan) return 0;
    if (a_nan) return 1;
    if (b_nan) return -1;
    if (a < b) return -1;
    if (a > b) return 1;
    /* Equal values: distinguish -0.0 from +0.0 via bit pattern */
    uint32_t ua, ub;
    memcpy(&ua, &a, sizeof(ua));
    memcpy(&ub, &b, sizeof(ub));
    /* Sign bit is bit 31. -0.0 = 0x80000000, +0.0 = 0x00000000 */
    /* We want -0.0 (sign=1) before +0.0 (sign=0) */
    int sa = (ua >> 31) & 1;
    int sb = (ub >> 31) & 1;
    if (sa != sb) return sb - sa;
    return 0;
}

/* Stable merge sort */
static void merge(float *arr, float *tmp, int left, int mid, int right) {
    int i = left, j = mid, k = left;
    while (i < mid && j < right) {
        if (compare(arr[i], arr[j]) <= 0) {
            tmp[k++] = arr[i++];
        } else {
            tmp[k++] = arr[j++];
        }
    }
    while (i < mid) tmp[k++] = arr[i++];
    while (j < right) tmp[k++] = arr[j++];
    memcpy(arr + left, tmp + left, (right - left) * sizeof(float));
}

static void merge_sort(float *arr, float *tmp, int left, int right) {
    if (right - left <= 1) return;
    int mid = left + (right - left) / 2;
    merge_sort(arr, tmp, left, mid);
    merge_sort(arr, tmp, mid, right);
    merge(arr, tmp, left, mid, right);
}

int main(void) {
    int n;
    if (fread(&n, sizeof(int), 1, stdin) != 1) return 1;

    float *arr = malloc(n * sizeof(float));
    float *tmp = malloc(n * sizeof(float));
    if (!arr || !tmp) return 1;

    if (fread(arr, sizeof(float), n, stdin) != (size_t)n) return 1;

    merge_sort(arr, tmp, 0, n);

    fwrite(arr, sizeof(float), n, stdout);

    free(arr);
    free(tmp);
    return 0;
}
