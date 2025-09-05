#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cuda_runtime.h>
#include "matrix_utils.h"

__global__ void matrixAddition(int rows, int cols, float* a, float* b, float* c) {
    // get global thread index
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    
    // matrix addition per thread
    if (row < rows && col < cols) {
        c[row * cols + col] = a[row * cols + col] + b[row * cols + col];
    }
}

int main() {
    int rows = 1024;
    int cols = 1024;

    // init input/output matrices in host
    float* host_a = (float*)malloc(rows * cols * sizeof(float));
    float* host_b = (float*)malloc(rows * cols * sizeof(float));
    float* host_c = (float*)malloc(rows * cols * sizeof(float));

    srand(5566);
    for (int i = 0; i < rows * cols; i++) {
        host_a[i] = (float)rand() / RAND_MAX;
        host_b[i] = (float)rand() / RAND_MAX;
    }
    
    // copy input/output matrices to device
    float* device_a;
    float* device_b;
    float* device_c;

    cudaMalloc(&device_a, rows * cols * sizeof(float));
    cudaMalloc(&device_b, rows * cols * sizeof(float));
    cudaMalloc(&device_c, rows * cols * sizeof(float));
    
    cudaMemcpy(device_a, host_a, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // matrix addition
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    matrixAddition<<<blocksPerGrid, threadsPerBlock>>>(rows, cols, device_a, device_b, device_c);
    cudaDeviceSynchronize();

    // copy output matrices to host
    cudaMemcpy(host_c, device_c, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // show result
    printMatrix("A", host_a, rows, cols);
    printMatrix("B", host_b, rows, cols);
    printMatrix("C", host_c, rows, cols);

    // clean up
    free(host_a);
    free(host_b);
    free(host_c);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return 0;
}