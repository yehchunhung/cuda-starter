#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cuda_runtime.h>
#include "../matrix_addition/matrix_utils.h"

__global__ void matrixTranspose(int rows, int cols, float* a) {
    // get global thread index
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    
    // matrix transpose per thread
    if (row < rows && col < cols) {
        a[col * rows + row] = a[row * cols + col];
    }
}

int main() {
    int rows = 1024;
    int cols = 512;
    
    // init input/output matrices in host
    float* host_a = (float*)malloc(rows * cols * sizeof(float));

    srand(5566);
    for (int i = 0; i < rows * cols; i++)
        host_a[i] = (float)rand() / RAND_MAX;

    printMatrix("A", host_a, rows, cols);

    // copy input/output matrices to device
    float* device_a;
    cudaMalloc(&device_a, rows * cols * sizeof(float));
    cudaMemcpy(device_a, host_a, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // matrix transpose
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    matrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(rows, cols, device_a);
    cudaDeviceSynchronize();

    // copy output matrices to host
    cudaMemcpy(host_a, device_a, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // show result
    printMatrix("A", host_a, cols, rows);

    // clean up
    free(host_a);
    cudaFree(device_a);

    return 0;
}