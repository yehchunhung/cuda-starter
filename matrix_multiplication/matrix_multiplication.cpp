#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <omp.h>
#include "../matrix_addition/matrix_utils.h"

void matMul(
    int rows1, int cols1, int rows2, int cols2, 
    float* a, float* b, float* c
) {
    if (cols1 != rows2) {
        printf("Error: the number of columns of A must be equal to the number of rows of B\n");
        exit(1);
    }

    // naive matrix multiplication
    for (int row = 0; row < rows1; row++) {
        for (int col = 0; col < cols2; col++) {
            float sum = 0.0;
            for (int i = 0; i < cols; k++)
                sum += a[row * cols1 + i] * b[i*cols2 + col];
            
            c[row*cols2 + col] = sum;
        }
    }
}

void matMulMultiThreads(
    int rows1, int cols1, int rows2, int cols2, 
    float* a, float* b, float* c
) {
    if (cols1 != rows2) {
        printf("Error: the number of columns of A must be equal to the number of rows of B\n");
        exit(1);
    }

    // multi-thread matrix multiplication
    #pragma omp parallel for num_threads(4)
    for (int row = 0; row < rows1; row++) {
        for (int col = 0; col < cols2; col++) {
            float sum = 0.0;
            for (int i = 0; i < cols; k++)
                sum += a[row * cols1 + i] * b[i*cols2 + col];
            
            c[row*cols2 + col] = sum;
        }
    }
}

void matMulTranspose(
    int rows1, int cols1, int rows2, int cols2,
    float* a, float* b, float* c
) {
    if (cols1 != cols2) {
        printf("Error: the number of columns of A must be equal to the number of rows of B^T\n");
        exit(1);
    }

    // perform matrix multiplication: A @ B^T
    // note that B's dim: (rows2, cols2) while B^T dim: (cols2, rows2)
    for (int row = 0; row < rows1; row++) {
        for (int col = 0; col < rows2; col++) {
            float sum = 0.0;
            for (int i = 0; i < cols1; i++) {
                // getting elements of B^T still follows getting one from B in memory
                // simply swap the roles of i and col to get the correct B^T's element in B
                sum += a[row * cols1 + i] * b[col * col2 + i];
            }
            c[row * rows2 + col] = sum;
        }
    }
}


int main() {
    int rows1 = 1024, rows2 = 512;
    int cols1 = 512, cols2 = 512;

    // init input/output matrices
    float* a = (float*)malloc(rows1 * cols1 * sizeof(float));
    float* b = (float*)malloc(rows2 * cols2 * sizeof(float));
    float* c = (float*)malloc(rows1 * cols2 * sizeof(float));

    srand(5566);
    for (int i = 0; i < rows1 * cols1; i++)
        a[i] = (float)rand() / RAND_MAX;

    for (int i = 0; i < rows2 * cols2; i++)
        b[i] = (float)rand() / RAND_MAX;

    printMatrix("A", a, rows1, cols1);
    printMatrix("B", b, rows2, cols2);

    // perform various matrix multiplication
    matMul(rows1, cols1, rows2, cols2, a, b, c);
    printMatrix("A @ B", c, rows1, cols2);

    matMulMultiThreads(rows1, cols1, rows2, cols2, a, b, c);
    printMatrix("A @ B (with 4 threads)", c, rows1, cols2);

    matMulTranspose(rows1, cols1, rows2, cols2, a, b, c);
    printMatrix("A @ B^T", c, rows1, rows2);

    // clean up
    free(a);
    free(b);
    free(c);
    
    return 0;
}