#include <stdio.h>
#include <algorithm>
#include "matrix_utils.h"

void printMatrix(const char* name, float* matrix, int rows, int cols, int maxPrint) {
    // helper function to small portion of matrix
    printf(
        "Matrix: %s (showing first %dx%d):\n", name, 
        std::min(rows, maxPrint), std::min(cols, maxPrint)
    );
    for (int row = 0; row < std::min(rows, maxPrint); row++) {
        for (int col = 0; col < std::min(cols, maxPrint); col++) {
            printf("%f ", matrix[row * cols + col]);
        }
        if (cols > maxPrint)
            printf("...");

        printf("\n");
    }
    if (rows > maxPrint)
        printf("...");
    printf("\n");
}