#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <cassert>
#include "../matrix_addition/matrix_utils.h"

#define TILE_SIZE 32

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

__global__ void matMulSharedMemory(
    int rows1, int cols1, int rows2, int cols2,
    float* a, float* b, float* c
) {
    assert(cols1 == rows2 && "Matrix dimensions unmatched for multiplication!");

    // init sub-matrices used in shared memory
    __shared__ float shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_b[TILE_SIZE][TILE_SIZE];

    // local indices on sub-matrices, shared_a and shared_b
    int tx = threadIdx.x, ty = threadIdx.y;

    // indices on output matrix c
    int row = ty + blockDim.y * blockIdx.y;
    int col = tx + blockDim.x * blockIdx.x;

    // loop over sub-matrices
    float sum = 0.0;
    for (int m = 0; m < (cols1 + TILE_SIZE - 1) / TILE_SIZE; m++) {
        // copy elements from original matrices to sub-matrices
        if (row < rows1 && m*TILE_SIZE + tx < cols1)
            shared_a[ty][tx] = a[row*cols1 + m*TILE_SIZE + tx];
        else
            shared_a[ty][tx] = 0.0;

        if (m*TILE_SIZE + ty < rows2 && col < cols2)
            shared_b[ty][tx] = b[(m*TILE_SIZE + ty)*cols2 + col];
        else
            shared_b[ty][tx] = 0.0;

        __syncthreads();
        
        // compute matmul per thread within a block
        for (int i = 0; i < TILE_SIZE; i++)
            sum += shared_a[ty][i] * shared_b[i][tx];

        __syncthreads();
    }

    // put the result to output matrix
    if (row < rows1 && col < cols2)
        c[row * cols2 + col] = sum;
}


__global__ void matMulTransposeSharedMemory(
    int rows1, int cols1, int rows2, int cols2,
    float* a, float* b, float* c
) {
    assert(cols1 == rows2 && "Matrix dimensions unmatched for multiplication!");

    // init sub-matrices used in shared memory
    // add one padding in shared_b to avoid bank conflict
    __shared__ float shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_b[TILE_SIZE][TILE_SIZE+1];

    // local indices on sub-matrices, shared_a and shared_b
    int tx = threadIdx.x, ty = threadIdx.y;

    // indices on output matrix c
    int row = ty + blockDim.y * blockIdx.y;
    int col = tx + blockDim.x * blockIdx.x;

    // loop over sub-matrices
    float sum = 0.0;
    for (int m = 0; m < (cols1 + TILE_SIZE - 1) / TILE_SIZE; m++) {
        // copy elements from original matrices to sub-matrices
        if (row < rows1 && m*TILE_SIZE + tx < cols1)
            shared_a[ty][tx] = a[row*cols1 + m*TILE_SIZE + tx];
        else
            shared_a[ty][tx] = 0.0;

        if (m*TILE_SIZE + ty < rows2 && col < cols2)
            // shared_b[tx][ty] is transpose of shared_b[ty][tx]
            shared_b[tx][ty] = b[(m*TILE_SIZE + ty)*cols2 + col];
        else
            shared_b[tx][ty] = 0.0;

        __syncthreads();
        
        // threads are placed row-wise along a column
        // encounter bank conflict issue leading to slow serialization
        for (int i = 0; i < TILE_SIZE; i++)
            sum += shared_a[ty][i] * shared_b[tx][i];

        __syncthreads();
    }

    // put the result to output matrix
    if (row < rows1 && col < cols2)
        c[row * cols2 + col] = sum;
}


int main() {
    int rows1 = 2048, rows2 = 2048;
    int cols1 = 2048, cols2 = 2048;
    double start, end;

    // init input/output matrices in CPU
    float* host_a = (float*)malloc(rows1 * cols1 * sizeof(float));
    float* host_b = (float*)malloc(rows2 * cols2 * sizeof(float));
    float* host_c = (float*)malloc(rows1 * cols2 * sizeof(float));

    srand(5566);
    for (int i = 0; i < rows1 * cols1; i++)
        host_a[i] = (float)rand() / RAND_MAX;

    for (int i = 0; i < rows2 * cols2; i++)
        host_b[i] = (float)rand() / RAND_MAX;

    printMatrix("A", host_a, rows1, cols1);
    printMatrix("B", host_b, rows2, cols2);

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
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(
        (cols2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows1 + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // standard matmul
    start = get_time_ms();
    matMul<<<blocksPerGrid, threadsPerBlock>>>(rows1, cols1, rows2, cols2, device_a, device_b, device_c);
    cudaDeviceSynchronize();
    end = get_time_ms();
    cudaMemcpy(host_c, device_c, rows1 * cols2 * sizeof(float), cudaMemcpyDeviceToHost);
    printMatrix("A @ B", host_c, rows1, cols2);
    printf("Time: %.6f ms\n\n", end - start);

    // matmul with shared memory
    start = get_time_ms();
    matMulSharedMemory<<<blocksPerGrid, threadsPerBlock>>>(rows1, cols1, rows2, cols2, device_a, device_b, device_c);
    cudaDeviceSynchronize();
    end = get_time_ms();
    cudaMemcpy(host_c, device_c, rows1 * cols2 * sizeof(float), cudaMemcpyDeviceToHost);
    printMatrix("A @ B (using shared memory and memory coalescing)", host_c, rows1, cols2);
    printf("Time: %.6f ms\n\n", end - start);

    // matmul with shared memory
    start = get_time_ms();
    matMulTransposeSharedMemory<<<blocksPerGrid, threadsPerBlock>>>(rows1, cols1, rows2, cols2, device_a, device_b, device_c);
    cudaDeviceSynchronize();
    end = get_time_ms();
    cudaMemcpy(host_c, device_c, rows1 * cols2 * sizeof(float), cudaMemcpyDeviceToHost);
    printMatrix("A @ B (using shared memory and avoid bank conflict)", host_c, rows1, cols2);
    printf("Time: %.6f ms\n\n", end - start);

    // clean up
    free(host_a);
    free(host_b);
    free(host_c);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return 0;
}