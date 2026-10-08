/*
 * matmul_cpu.c - CPU baseline for square matrix multiplication (C = A * B).
 *
 * Compares two single-threaded loop orders:
 *   ijk: the textbook version. The inner loop walks down a column of B,
 *        jumping n floats in memory each step, so it misses the cache often.
 *   ikj: same math, loops reordered. The inner loop walks along a row of B
 *        and a row of C, touching memory sequentially, so the cache helps.
 *
 * Build: gcc -O2 -o matmul_cpu src/matmul_cpu.c -lm
 * Run:   ./matmul_cpu            (default sizes 256 512 1024)
 *        ./matmul_cpu 128 256    (custom sizes)
 * Output is CSV: n,version,ms,gflops
 */
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Fill an n x n matrix with random values in [-0.5, 0.5]. */
static void fill_random(float *m, int n) {
    for (long i = 0; i < (long)n * n; i++) {
        m[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    }
}

/* Textbook order: for each output element, take a dot product. */
static void matmul_ijk(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j]; /* B accessed by column: strided */
            }
            C[i * n + j] = sum;
        }
    }
}

/* Cache-friendly order: add a scaled row of B into a row of C. */
static void matmul_ikj(const float *A, const float *B, float *C, int n) {
    memset(C, 0, sizeof(float) * (size_t)n * n);
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float a = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += a * B[k * n + j]; /* B and C accessed by row: sequential */
            }
        }
    }
}

/* Largest element-wise difference, relative to the largest reference value. */
static double max_rel_error(const float *ref, const float *got, int n) {
    double max_diff = 0.0, max_ref = 0.0;
    for (long i = 0; i < (long)n * n; i++) {
        double d = fabs((double)ref[i] - (double)got[i]);
        if (d > max_diff) max_diff = d;
        if (fabs(ref[i]) > max_ref) max_ref = fabs(ref[i]);
    }
    return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}

static double gflops(int n, double seconds) {
    return 2.0 * (double)n * n * n / (seconds * 1e9); /* n^3 multiply-adds = 2n^3 FLOPs */
}

int main(int argc, char **argv) {
    int default_sizes[] = {256, 512, 1024};
    int num_sizes = argc > 1 ? argc - 1 : 3;

    srand(42);
    printf("n,version,ms,gflops\n");

    for (int s = 0; s < num_sizes; s++) {
        int n = argc > 1 ? atoi(argv[s + 1]) : default_sizes[s];
        if (n <= 0) {
            fprintf(stderr, "Invalid size: %s\n", argv[s + 1]);
            return 1;
        }
        size_t bytes = sizeof(float) * (size_t)n * n;
        float *A = malloc(bytes), *B = malloc(bytes);
        float *C1 = malloc(bytes), *C2 = malloc(bytes);
        if (!A || !B || !C1 || !C2) {
            fprintf(stderr, "Out of memory at n=%d\n", n);
            return 1;
        }
        fill_random(A, n);
        fill_random(B, n);

        double t0 = now_sec();
        matmul_ijk(A, B, C1, n);
        double t_ijk = now_sec() - t0;

        t0 = now_sec();
        matmul_ikj(A, B, C2, n);
        double t_ikj = now_sec() - t0;

        double err = max_rel_error(C1, C2, n);
        if (err > 1e-3) {
            fprintf(stderr, "Mismatch at n=%d: relative error %.2e\n", n, err);
            return 1;
        }

        printf("%d,cpu_ijk,%.2f,%.3f\n", n, t_ijk * 1e3, gflops(n, t_ijk));
        printf("%d,cpu_ikj,%.2f,%.3f\n", n, t_ikj * 1e3, gflops(n, t_ikj));
        fflush(stdout);

        free(A); free(B); free(C1); free(C2);
    }
    return 0;
}
