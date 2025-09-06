#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "../matrix_addition/matrix_utils.h"

void matrixTranspose(int rows, int cols, float* a) {
    // naive matrix transpose
    // create a temporary matrix to store the transpose
    float* temp = (float*)malloc(rows * cols * sizeof(float));

    // transpose the matrix
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            temp[col * rows + row] = a[row * cols + col];
        }
    }
    // copy the transpose back to the original matrix
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            a[row * cols + col] = temp[row * cols + col];
        }
    }
    // clean up
    free(temp);
}

int main() {
    int rows = 1024;
    int cols = 512;

    // init input/output matrices in host
    float* a = (float*)malloc(rows * cols * sizeof(float));

    srand(5566);
    for (int i = 0; i < rows * cols; i++)
        a[i] = (float)rand() / RAND_MAX;

    // matrix transpose
    matrixTranspose(rows, cols, a);

    // show result
    printMatrix("A", a, rows, cols);

    // clean up
    free(a);

    return 0;
}