#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

void vectorAddition(int rows, int cols, float* a, float* b, float* c) {
    // naive vector addition
    for (int i = 0; i < rows * cols; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int rows = 1024;
    int cols = 1024;

    // init input/output matrices
    float* a = (float*)malloc(rows * cols * sizeof(float));
    float* b = (float*)malloc(rows * cols * sizeof(float));
    float* c = (float*)malloc(rows * cols * sizeof(float));

    srand(5566);
    for (int i = 0; i < rows * cols; i++) {
        a[i] = (float)rand() / RAND_MAX;
        b[i] = (float)rand() / RAND_MAX;
    }

    // vector addition
    vectorAddition(rows, cols, a, b, c);

    // show result
    for (int i = 0; i < rows * cols; i++) {
        printf("%f + %f = %f\n", a[i], b[i], c[i]);
    }

    // clean up
    free(a);
    free(b);
    free(c);

    return 0;
}