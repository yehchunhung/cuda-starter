#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "matrix_utils.h"

void matrixAddition(int rows, int cols, float* a, float* b, float* c) {
    // naive matrix addition
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            c[row * cols + col] = a[row * cols + col] + b[row * cols + col];
        }
    }
}

int main() {
    int rows = 1024;
    int cols = 1024;

    // init input/output matrices in host
    float* a = (float*)malloc(rows * cols * sizeof(float));
    float* b = (float*)malloc(rows * cols * sizeof(float));
    float* c = (float*)malloc(rows * cols * sizeof(float));

    srand(5566);
    for (int i = 0; i < rows * cols; i++) {
        a[i] = (float)rand() / RAND_MAX;
        b[i] = (float)rand() / RAND_MAX;
    }

    // matrix addition
    matrixAddition(rows, cols, a, b, c);

    // show result
    printMatrix("A", a, rows, cols);
    printMatrix("B", b, rows, cols);
    printMatrix("C", c, rows, cols);

    // clean up
    free(a);
    free(b);
    free(c);

    return 0;
}