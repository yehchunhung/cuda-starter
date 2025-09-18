#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <omp.h>
#include "../matrix_addition/matrix_utils.h"

void matMulNaive(
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
            for (int i = 0; i < cols1; i++)
                c[row*cols2 + col] += a[row * cols1 + i] * b[i*cols2 + col];
        }
    }
}

void matMulCoalesced(
    int rows1, int cols1, int rows2, int cols2, 
    float* a, float* b, float* c
) {
    if (cols1 != rows2) {
        printf("Error: the number of columns of A must be equal to the number of rows of B\n");
        exit(1);
    }

    // matrix multiplication to achieve lower cache miss
    // loop reordering to access consecutive elements from b in the inner most loop
    for (int row = 0; row < rows1; row++) {
        for (int i = 0; i < cols1; i++) {
            for (int col = 0; col < cols2; col++)
                c[row*cols2 + col] += a[row * cols1 + i] * b[i * cols2 + col];
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
    #pragma omp parallel for
    for (int row = 0; row < rows1; row++) {
        for (int i = 0; i < cols1; i++) {
            // cache elements for reuse
            const float a_val = a[row * cols1 + i];
            const float* b_row = &b[i * cols2];

            #pragma omp simd
            for (int col = 0; col < cols2; col++)
                c[row*cols2 + col] += a_val * b_row[col];
        }
    }
}

void matMulTransposeMultiThread(
    int rows1, int cols1, int rows2, int cols2,
    float* a, float* b, float* c
) {
    if (cols1 != rows2) {
        printf("Error: the number of columns of A must be equal to the number of rows of B^T\n");
        exit(1);
    }

    // perform matrix multiplication: A @ B^T
    // note that B's dim: (rows2, cols2) while B^T dim: (cols2, rows2)
    #pragma omp parallel for
    for (int row = 0; row < rows1; row++) {
        for (int col = 0; col < rows2; col++) {
            const float* a_row = &a[row * cols1];
            const float* b_row = &b[col * cols2];

            #pragma omp simd // fused multiply-add and vectorization
            for (int i = 0; i < cols1; i++)
                // getting elements of B^T still follows getting one from B in memory
                // simply swap the roles of i and col to get the correct B^T's element in B
                c[row * rows2 + col] += a_row[i] * b_row[i];
        }
    }
}

int main() {
    int rows1 = 1024, rows2 = 512;
    int cols1 = 512, cols2 = 512;
    double start, end;

    // init input/output matrices
    float* a = (float*)malloc(rows1 * cols1 * sizeof(float));
    float* b = (float*)malloc(rows2 * cols2 * sizeof(float));
    float* c = (float*)calloc(rows1 * cols2, sizeof(float));

    srand(5566);
    for (int i = 0; i < rows1 * cols1; i++)
        a[i] = (float)rand() / RAND_MAX;

    for (int i = 0; i < rows2 * cols2; i++)
        b[i] = (float)rand() / RAND_MAX;

    printMatrix("A", a, rows1, cols1);
    printMatrix("B", b, rows2, cols2);

    // perform various matrix multiplication
    start = get_time_ms();
    matMulNaive(rows1, cols1, rows2, cols2, a, b, c);
    end = get_time_ms();
    printMatrix("A @ B", c, rows1, cols2);
    printf("Time: %.6f ms\n\n", end - start);

    // perform faster matmul by loop reordering
    c = (float*)calloc(rows1 * cols2, sizeof(float)); // reset
    start = get_time_ms();
    matMulCoalesced(rows1, cols1, rows2, cols2, a, b, c);
    end = get_time_ms();
    printMatrix("A @ B (loop reordering to achieve lower cache miss)", c, rows1, cols2);
    printf("Time: %.6f ms\n\n", end - start);

    // consider multi threads as well
    c = (float*)calloc(rows1 * cols2, sizeof(float)); // reset
    start = get_time_ms();
    matMulMultiThreads(rows1, cols1, rows2, cols2, a, b, c);
    end = get_time_ms();
    printMatrix("A @ B (multi threads)", c, rows1, cols2);
    printf("Time: %.6f ms\n\n", end - start);

    // A @ B^T by considering the above features
    c = (float*)calloc(rows1 * cols2, sizeof(float)); // reset
    start = get_time_ms();
    matMulTransposeMultiThread(rows1, cols1, rows2, cols2, a, b, c);
    end = get_time_ms();
    printMatrix("A @ B^T (multi threads)", c, rows1, rows2);
    printf("Time: %.6f ms\n\n", end - start);

    // clean up
    free(a);
    free(b);
    free(c);
    
    return 0;
}