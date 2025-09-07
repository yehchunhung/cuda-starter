#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <cassert>
#include "../matrix_addition/matrix_utils.h"

__global__ void matMul(
    int rows1, int cols1, int rows2, int cols2,
    float* a, float* b, float* c
) {
    // exit(1) unsupported in GPU
    assert(cols1 == rows2 && "Matrix dimensions unmatched for multiplication!");

    // get global thread index
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    // matrix multiplication per thread
    if (row < rows1 && col < cols2) {
        float sum = 0.0;
        for (int i = 0; i < cols1; i++)
            sum += a[row * cols1 + i] * b[i * cols2 + col];

        c[row * cols2 + col] = sum;
    }
}

int main() {
    int rows1 = 1024, rows2 = 512;
    int cols1 = 512, cols2 = 512;

    // init input/output matrices in CPU
    float* host_a = (float*)malloc(rows1 * cols1 * sizeof(float));
    float* host_b = (float*)malloc(rows2 * cols2 * sizeof(float));
    float* host_c = (float*)malloc(rows1 * cols2 * sizeof(float));

    srand(5566);
    for (int i = 0; i < rows1 * cols1; i++)
        host_a[i] = (float)rand() / RAND_MAX;

    for (int i = 0; i < rows2 * cols2; i++)
        host_b[i] = (float)rand() / RAND_MAX;

    // init input/output matrics in GPU
    float* device_a;
    float* device_b;
    float* device_c;
    cudaMalloc(&device_a, rows1 * cols1 * sizeof(float));
    cudaMalloc(&device_b, rows2 * cols2 * sizeof(float));
    cudaMalloc(&device_c, rows1 * cols2 * sizeof(float));

    // copy data from CPU to GPU
    cudaMemcpy(device_a, host_a, rows1 * cols1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, rows2 * cols2 * sizeof(float), cudaMemcpyHostToDevice);

    // perform matrix multiplication
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (cols2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows1 + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    matMul<<<blocksPerGrid, threadsPerBlock>>>(rows1, cols1, rows2, cols2, device_a, device_b, device_c);
    cudaDeviceSynchronize();
    
    // copy result back to CPU
    cudaMemcpy(host_c, device_c, rows1 * cols2 * sizeof(float), cudaMemcpyDeviceToHost);

    // show result
    printMatrix("A", host_a, rows1, cols1);
    printMatrix("B", host_b, rows1, cols1);
    printMatrix("A @ B", host_c, rows1, cols2);

    // clean up
    free(host_a);
    free(host_b);
    free(host_c);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return 0;
}