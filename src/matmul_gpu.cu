/*
 * matmul_gpu.cu - GPU matrix multiplication benchmark (C = A * B, square, float32).
 *
 * Versions compared:
 *   cpu_ikj    single-threaded CPU reference (also used to check correctness)
 *   gpu_naive  one thread per output element, reads A and B from global memory
 *   gpu_tiled  same thread layout, but each block loads TILE x TILE tiles of A and B
 *              into shared memory and reuses them, cutting global memory reads by ~TILE x
 *   cublas     NVIDIA's tuned library, as the upper bound to compare against
 *
 * Build: nvcc -O3 -arch=native -o matmul_gpu src/matmul_gpu.cu -lcublas
 * Run:   ./matmul_gpu                 (default sizes 256 512 1024 2048)
 *        ./matmul_gpu 512 4096        (custom sizes; CPU check is skipped above 2048)
 * Output is CSV: n,version,ms,gflops,speedup_vs_cpu,max_rel_err
 * Kernel times exclude host<->device copies; copy time is reported separately on stderr.
 */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TILE 16          /* tile width; a block has TILE x TILE = 256 threads */
#define REPEATS 5        /* timed runs per kernel, averaged */
#define CPU_CHECK_MAX 2048

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                        \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st_ = (call);                                            \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_,             \
                    __FILE__, __LINE__);                                        \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

/* ---------------------------------------------------------------- kernels */

/* Each thread computes C[row][col] as a dot product of row `row` of A and column
 * `col` of B. Every multiply-add needs two global memory loads, so this kernel is
 * limited by memory bandwidth, not arithmetic. */
__global__ void matmul_naive(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

/* Same output layout as the naive kernel, but the block walks across A and down B
 * one TILE x TILE tile at a time. Each thread loads ONE element of each tile into
 * shared memory, then all threads in the block reuse the whole tile. Each value
 * loaded from global memory is used TILE times instead of once. */
__global__ void matmul_tiled(const float *A, const float *B, float *C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float sum = 0.0f;

    int num_tiles = (n + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;
        /* Pad with zeros when n is not a multiple of TILE. */
        As[ty][tx] = (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;
        __syncthreads(); /* wait until the whole tile is loaded */

        for (int k = 0; k < TILE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads(); /* wait until everyone is done before overwriting the tile */
    }
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

/* ------------------------------------------------------------ host helpers */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fill_random(float *m, int n) {
    for (long i = 0; i < (long)n * n; i++) {
        m[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    }
}

static void matmul_cpu_ikj(const float *A, const float *B, float *C, int n) {
    memset(C, 0, sizeof(float) * (size_t)n * n);
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float a = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += a * B[k * n + j];
            }
        }
    }
}

static double max_rel_error(const float *ref, const float *got, int n) {
    double max_diff = 0.0, max_ref = 0.0;
    for (long i = 0; i < (long)n * n; i++) {
        double d = fabs((double)ref[i] - (double)got[i]);
        if (d > max_diff) max_diff = d;
        if (fabs(ref[i]) > max_ref) max_ref = fabs(ref[i]);
    }
    return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}

static double gflops(int n, double ms) {
    return 2.0 * (double)n * n * n / (ms * 1e-3 * 1e9);
}

typedef enum { NAIVE, TILED, CUBLAS } Version;

/* Launch one version once. */
static void launch(Version v, const float *dA, const float *dB, float *dC, int n,
                   cublasHandle_t handle) {
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    if (v == NAIVE) {
        matmul_naive<<<grid, block>>>(dA, dB, dC, n);
    } else if (v == TILED) {
        matmul_tiled<<<grid, block>>>(dA, dB, dC, n);
    } else {
        /* cuBLAS is column-major. A row-major matrix read as column-major is its
         * transpose, so computing B^T * A^T in cuBLAS yields (A*B)^T in column-major,
         * which is exactly A*B in row-major. */
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                 &alpha, dB, n, dA, n, &beta, dC, n));
    }
    CUDA_CHECK(cudaGetLastError());
}

/* Warm up once, then average REPEATS timed runs using CUDA events. */
static float time_version(Version v, const float *dA, const float *dB, float *dC, int n,
                          cublasHandle_t handle) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    launch(v, dA, dB, dC, n, handle); /* warm-up: first launch includes setup cost */
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < REPEATS; r++) {
        launch(v, dA, dB, dC, n, handle);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / REPEATS;
}

/* ------------------------------------------------------------------- main */

int main(int argc, char **argv) {
    int default_sizes[] = {256, 512, 1024, 2048};
    int num_sizes = argc > 1 ? argc - 1 : 4;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "GPU: %s (compute capability %d.%d)\n", prop.name, prop.major,
            prop.minor);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    srand(42);
    printf("n,version,ms,gflops,speedup_vs_cpu,max_rel_err\n");

    for (int s = 0; s < num_sizes; s++) {
        int n = argc > 1 ? atoi(argv[s + 1]) : default_sizes[s];
        if (n <= 0) {
            fprintf(stderr, "Invalid size: %s\n", argv[s + 1]);
            return 1;
        }
        size_t bytes = sizeof(float) * (size_t)n * n;

        float *A = (float *)malloc(bytes), *B = (float *)malloc(bytes);
        float *C_ref = (float *)malloc(bytes), *C_gpu = (float *)malloc(bytes);
        if (!A || !B || !C_ref || !C_gpu) {
            fprintf(stderr, "Out of host memory at n=%d\n", n);
            return 1;
        }
        fill_random(A, n);
        fill_random(B, n);

        /* CPU reference (skipped for large n because it takes too long). */
        int check = n <= CPU_CHECK_MAX;
        double cpu_ms = -1.0;
        if (check) {
            double t0 = now_sec();
            matmul_cpu_ikj(A, B, C_ref, n);
            cpu_ms = (now_sec() - t0) * 1e3;
            printf("%d,cpu_ikj,%.3f,%.2f,1.00,0\n", n, cpu_ms, gflops(n, cpu_ms));
        }

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, bytes));
        CUDA_CHECK(cudaMalloc(&dC, bytes));

        double t0 = now_sec();
        CUDA_CHECK(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        fprintf(stderr, "n=%d: host-to-device copy of A and B took %.3f ms\n", n,
                (now_sec() - t0) * 1e3);

        const char *names[] = {"gpu_naive", "gpu_tiled", "cublas"};
        Version versions[] = {NAIVE, TILED, CUBLAS};

        for (int v = 0; v < 3; v++) {
            CUDA_CHECK(cudaMemset(dC, 0, bytes));
            float ms = time_version(versions[v], dA, dB, dC, n, handle);
            CUDA_CHECK(cudaMemcpy(C_gpu, dC, bytes, cudaMemcpyDeviceToHost));

            double err = check ? max_rel_error(C_ref, C_gpu, n) : -1.0;
            if (check && err > 1e-3) {
                fprintf(stderr, "FAILED: %s at n=%d has relative error %.2e\n",
                        names[v], n, err);
                return 1;
            }
            double speedup = check ? cpu_ms / ms : -1.0;
            printf("%d,%s,%.3f,%.2f,%.2f,%.2e\n", n, names[v], ms, gflops(n, ms),
                   speedup, err);
            fflush(stdout);
        }

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
        free(A); free(B); free(C_ref); free(C_gpu);
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    fprintf(stderr, "All GPU results matched the CPU reference.\n");
    return 0;
}
