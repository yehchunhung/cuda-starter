#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cuda_runtime.h>

__global__ void vectorAddition(int rows, int cols, float* a, float* b, float* c) {
    // get global thread index
    int i = threadIdx.x + threadIdx.y * blockDim.x;

    // vector addition per thread
    if (idx < rows * cols) {
        c[i] = a[i] + b[i];
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

    cudaMemcpy(device_a, host_a, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // vector addition
    int threadsPerBlock = 16;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddition<<<blocksPerGrid, threadsPerBlock>>>(rows, cols, device_a, device_b, device_c);
    cudaDeviceSynchronize(); // wait for the kernel to finish

    // copy output matrices to host
    cudaMemcpy(host_c, device_c, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // show result
    for (int i = 0; i < rows * cols; i++) {
        printf("%f + %f = %f\n", host_a[i], host_b[i], host_c[i]);
    }

    // clean up
    free(host_a);
    free(host_b);
    free(host_c);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return 0;
}