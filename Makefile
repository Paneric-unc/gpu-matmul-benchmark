# Build both benchmarks. On Colab or any machine with the CUDA toolkit: `make`
# If -arch=native fails on an older toolkit, use -arch=sm_75 (T4) or sm_86 (A6000).
NVCC_ARCH ?= -arch=native

all: matmul_cpu matmul_gpu

matmul_cpu: src/matmul_cpu.c
	gcc -O2 -Wall -Wextra -o $@ $< -lm

matmul_gpu: src/matmul_gpu.cu
	nvcc -O3 $(NVCC_ARCH) -o $@ $< -lcublas

clean:
	rm -f matmul_cpu matmul_gpu *.csv *.png

.PHONY: all clean
