# Matrix Multiplication (CUDA and CPU)

This module implements three GPU kernels plus CPU baselines for matrix multiplication. Tile size is `TILE_SIZE = 32`.

### Implementation

- **matMul (naive global-memory)**: Each thread computes one `C[row, col]` directly from global memory with no tiling/coalescing. Simple and correct but bandwidth-inefficient.

- **matMulSharedMemory (tiled, coalesced)**: Uses two `32x32` shared-memory tiles (`shared_a`, `shared_b`). Threads cooperatively load a tile of A and B from global memory (coalesced), synchronize, then accumulate partial results over the tile loop. This reduces global memory traffic and improves performance.

- **matMulTransposeSharedMemory (tiled with transposed B tile + padding)**:
  - Loads the `B` tile into shared memory transposed, so later accesses are along rows rather than columns of the tile.
  - Declares `shared_b` with an extra column to avoid shared-memory bank conflicts:

```cpp
__shared__ float shared_a[TILE_SIZE][TILE_SIZE];
__shared__ float shared_b[TILE_SIZE][TILE_SIZE + 1]; // padding to avoid bank conflicts
```

### Bank conflict issue and fix

#### Problem
Without padding, accessing `shared_b` column-wise after transposing (e.g., `shared_b[tx][i]`) causes threads in the same warp to hit the same shared-memory bank when `TILE_SIZE` is a multiple of the number of banks (32). This leads to serialization and reduced throughput.

#### Why it happens
A 2D array laid out in row-major with width `TILE_SIZE` gives a stride equal to `TILE_SIZE`. With `TILE_SIZE = 32`, the address stride modulo the bank count (32) is 0, so many threads collide on the same bank during column-wise access.


#### Solution
Pad the second dimension by 1: `TILE_SIZE + 1`. This changes the row stride to 33 floats, breaking the harmful alignment so each thread in a warp maps to a different bank. The padding slightly increases shared memory usage but removes the conflict and improves performance.


### Build and run

```bash
# From matrix_multiplication/
make cuda && ./matrix_multiplication_cuda
make cpp && ./matrix_multiplication_cpp
```
