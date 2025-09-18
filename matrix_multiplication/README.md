# Matrix Multiplication Implementations

The directory includes multiple implementations showcasing the progression from naive matrix multiplication to optimized CUDA kernels with shared memory and proper memory access patterns.

## Implementations

### C++ Implementation (`matrix_multiplication.cpp`)

1. **`matMulNaive`** - Basic sequential matrix multiplication
   - Standard triple-nested loop implementation (row-col-i order)
   - Time complexity: O(n³)
   - Suffers from <span style="color:red;">**poor cache utilization**</span>

2. **`matMulOptimized`** - Cache-efficient loop reordering
   - Uses <span style="color:yellow;">**row-i-col**</span> loop order instead of row-col-i
   - <span style="color:yellow;">Dramatically improves cache hit rates</span>
   - Same O(n³) complexity but much faster execution

3. **`matMulMultiThread`** - OpenMP parallelized version with loop reordering
   - Combines cache-efficient loop ordering with multi-threading
   - Uses `#pragma omp parallel for schedule(static)` for parallelism

4. **`matMulWithTransposeMultiThread`** - Optimized for transposed matrix B
   - Computes A × B^T without explicitly transposing B
   - Improves cache locality by accessing B in row-major order

To learn more about details, you may open up the toggle blocks.

<details>
<summary>Why Loop Reordering Makes Matrix Multiplication SUPER Fast?</summary>

## Why Loop Reordering Makes Matrix Multiplication SUPER Fast (C++)

### The Problem with Naive Implementation

The naive approach uses row-col-i loop ordering:
```cpp
for (row)     // Pick output row
  for (col)   // Pick output column
    for (i)     // Compute dot product
      temp += a[row*cols1 + i] * b[i*cols2 + col];
```

This causes **terrible cache performance** because:
- When accessing `b[i*cols2 + col]` with incrementing `i`, we jump `cols2` positions in memory
- For a 1024×1024 matrix, that's 4KB jumps between each access!
- The CPU cache loads entire cache lines (64 bytes) but we only use 4 bytes
- Result: Mostly cache misses → slow memory access from RAM

### The Cache-Efficient Solution

The optimized version uses row-i-col loop ordering:
```cpp
for (row)     // Pick output row
  for (i)       // Fix one element of A
    for (col)   // Access B row sequentially
      c[row*cols2 + col] += a[row*cols1 + i] * b[i*cols2 + col];
```

This is **dramatically faster** because:
- Matrix B is accessed sequentially (`b[i*cols2 + 0], b[i*cols2 + 1], ...`)
- Adjacent elements in memory → perfect cache utilization
- CPU prefetcher can predict and preload the next data
- Result: Mostly cache hits → fast access from CPU cache

### Understanding Computer Memory Hierarchy

Think of it like a desk setup:
1. **CPU Registers** = Your hands (8-16 slots, instant access)
2. **L1 Cache** = Your desk (32-64 KB, ~4 cycles)
3. **L2 Cache** = Your bookshelf (256-512 KB, ~12 cycles)
4. **L3 Cache** = Your room (8-32 MB, ~40 cycles)
5. **RAM** = The library (GBs, ~100-300 cycles)

When you access data, the CPU doesn't just grab one number - it grabs an entire "cache line" (typically 64 bytes = 16 floats).

### Visual Example: Memory Access Patterns

**Naive (row-col-i):** Accessing B column-wise
```
Iteration 1: B[0,0] → Load cache line containing B[0,0-15]
Iteration 2: B[1,0] → Cache miss! (1024 positions away)
Iteration 3: B[2,0] → Cache miss! (another 1024 away)
...
Result: ~90% cache misses
```

**Optimized (row-i-col):** Accessing B row-wise
```
Iteration 1: B[i,0] → Load cache line containing B[i,0-15]
Iteration 2: B[i,1] → Cache hit! (already loaded)
Iteration 3: B[i,2] → Cache hit! (already loaded)
...
Result: ~95% cache hits
```

### Performance Impact

The difference is staggering:
- **Cache miss penalty**: 100-300 CPU cycles
- **Cache hit**: 1-4 CPU cycles
- **Speedup**: Often 5-10× faster just from loop reordering!

This optimization doesn't change the algorithm complexity (still O(n³)), but it makes each operation much faster by working with the hardware's memory system instead of against it.
</details>

<details>
<summary>SIMD Vectorization with OpenMP</summary>

## SIMD Vectorization with OpenMP (C++)

### What is `#pragma omp simd`?

SIMD (Single Instruction, Multiple Data) allows the CPU to process multiple data elements with a single instruction. Modern CPUs have vector registers that can hold multiple values:
- **AVX**: 256-bit registers (8 floats or 4 doubles)
- **AVX-512**: 512-bit registers (16 floats or 8 doubles)

### How SIMD Works - The Crayon Analogy

Imagine coloring 8 crayons:
- **Without SIMD**: Pick up one crayon, color it, put it down. Repeat 8 times.
- **With SIMD**: Pick up 8 crayons at once, color all 8 in one motion!

### Using SIMD in Matrix Multiplication

```cpp
// Without SIMD - processes one element at a time
for (int col = 0; col < cols2; col++)
    c[row*cols2 + col] += a[row*cols1 + i] * b[i*cols2 + col];

// With SIMD - processes 8 elements simultaneously (AVX)
#pragma omp simd
for (int col = 0; col < cols2; col++)
    c[row*cols2 + col] += a[row*cols1 + i] * b[i*cols2 + col];
```

The compiler generates vector instructions that:
1. Load 8 consecutive values from B into a vector register
2. Broadcast the single value from A to all lanes
3. Multiply all 8 pairs simultaneously
4. Add all 8 results to C simultaneously

### Combining Parallelization Techniques

You can combine multiple levels of parallelism:

```cpp
#pragma omp parallel for schedule(static)  // Thread-level parallelism
for (int row = 0; row < rows1; row++) {
    for (int i = 0; i < cols1; i++) {
        #pragma omp simd                   // Data-level parallelism
        for (int col = 0; col < cols2; col++)
            c[row*cols2 + col] += a[row*cols1 + i] * b[i*cols2 + col];
    }
}
```

This gives you:
- **Thread-level**: Multiple cores working on different rows
- **Data-level**: Each core processing 8-16 columns at once

### SIMD Best Practices

**Good candidates for SIMD:**
- Simple arithmetic operations
- Independent loop iterations
- Contiguous memory access
- No complex branching

**Poor candidates for SIMD:**
```cpp
// BAD: Loop-carried dependency
#pragma omp simd  // Won't vectorize effectively
for (int i = 1; i < n; i++)
    c[i] = c[i-1] + a[i];  // Each iteration depends on previous

// BAD: Complex branching
#pragma omp simd
for (int i = 0; i < n; i++)
    if (complex_condition(i))  // Divergent branching hurts SIMD
        c[i] = expensive_op(a[i]);
```

### Performance Impact

SIMD can provide:
- **4-8× speedup** for float operations (AVX: 8 floats at once)
- **2-4× speedup** for double operations (AVX: 4 doubles at once)
- **8-16× speedup** with AVX-512 (if supported)

Combined with loop reordering and multi-threading, modern CPUs can achieve remarkable performance on matrix multiplication through:
1. **Cache efficiency** (loop reordering for sequential access)
2. **Thread parallelism** (OpenMP parallel for)
3. **Data parallelism** (SIMD vectorization)

The key is that these optimizations work together multiplicatively - with 8 threads and 8-wide SIMD, you can process 64 elements simultaneously!

</details>


### CUDA Implementation (`matrix_multiplication.cu`)

1. **`matMul`** - Basic CUDA implementation
   - Each thread computes one element of the output matrix
   - Use <span style="color:yellow;">coalesced memory access (threadIdx.x for columns)</span>
   - Global memory access only

2. **`matMulNotCoalesced`** - Deliberately non-coalesced version
   - Swap thread indexing (threadIdx.x for rows, threadIdx.y for columns)
   - Demonstrate performance impact of <span style="color:red;">**non-coalesced memory access**</span>
   - Significantly slower due to strided memory access patterns

3. **`matMulSharedMemory`** - Tiled matrix multiplication with shared memory
   - Use <span style="color:yellow;">**shared memory**</span> to reduce global memory accesses
   - Implement tiling with `TILE_SIZE = 32`
   - Each block loads tiles into shared memory and computes partial results
   - <span style="color:yellow;">**No shared memory bank conflicts**</span> - optimal performance

4. **`matMulTransposeSharedMemory`** - Attempted optimization with transpose
   - Transpose matrix B during loading to shared memory
   - **UPDATE**: Now fixed with <span style="color:yellow;">padding</span> to avoid bank conflicts
   - Uses `__shared__ float shared_b[TILE_SIZE][TILE_SIZE+1]` for conflict-free access

To learn more about details, you may open up the toggle blocks.

<details>
<summary>Shared Memory Bank Conflicts</summary>

## Key Finding: Shared Memory Bank Conflicts (CUDA)

### The Problem

The original `matMulTransposeSharedMemory` function was slower than `matMulSharedMemory` due to **shared memory bank conflicts**.

### Root Cause Analysis

#### Shared Memory Architecture
- CUDA shared memory is divided into 32 banks
- Each bank is 4 bytes wide
- Consecutive 4-byte words map to consecutive banks
- Bank assignment: `bank_number = (byte_address / 4) % 32`

#### In `matMulSharedMemory` (No Conflicts)
```cuda
// Line 87: Each thread in a warp accesses different columns
temp += shared_a[ty][i] * shared_b[i][tx];
```
- Threads 0-31 access `shared_b[i][0]` through `shared_b[i][31]`
- These are consecutive memory locations → different banks
- **Result**: Parallel access, no conflicts

#### In Original `matMulTransposeSharedMemory` (Had Conflicts)
```cuda
// Line 139: Each thread in a warp accesses the same column
temp += shared_a[ty][i] * shared_b[tx][i];
```
- Threads 0-31 access `shared_b[0][i]` through `shared_b[31][i]`
- These addresses are separated by exactly 32 elements (TILE_SIZE)
- 32 elements = 128 bytes, which wraps around to the same bank
- **Result**: 32-way bank conflict, serialized access

### The Solution: Padding

Bank conflicts are resolved by adding padding to the shared memory array:

```cuda
// Original (with conflicts)
__shared__ float shared_b[TILE_SIZE][TILE_SIZE];

// Fixed (with padding)
__shared__ float shared_b[TILE_SIZE][TILE_SIZE+1];
```

#### How Padding Works

Without padding (`[32][32]`):
- Row 0: Elements 0-31 → Banks 0-31
- Row 1: Elements 32-63 → Banks 0-31 (wraps around)
- Row 2: Elements 64-95 → Banks 0-31 (wraps around)
- When accessing column `i`, all threads hit bank `i` → **Conflict!**

With padding (`[32][33]`):
- Row 0: Elements 0-31 → Banks 0-31
- Row 1: Elements 33-64 → Banks 1-0 (shifted by 1)
- Row 2: Elements 66-97 → Banks 2-1 (shifted by 2)
- When accessing column `i`, threads hit different banks → **No conflict!**

The extra column shifts each row's bank mapping by 1, ensuring threads in a warp access different banks.

## Performance Characteristics

### Memory Access Patterns

1. **Global Memory Coalescing**
   - Both `matMulSharedMemory` and `matMulTransposeSharedMemory` achieve coalesced global memory access
   - Threads in a warp access consecutive memory addresses when loading tiles

2. **Shared Memory Access**
   - `matMulSharedMemory`: Conflict-free by design
   - `matMulTransposeSharedMemory`: Requires padding to avoid conflicts

### Performance Ranking (Fastest to Slowest)

1. `matMulSharedMemory` - Optimal tiled implementation
2. `matMulTransposeSharedMemory` (with padding fix) - Similar performance after fix
3. `matMul` - Basic CUDA, coalesced but no shared memory
4. `matMulNotCoalesced` - Deliberately inefficient for comparison

</details>

## Key Lessons 

1. **Shared memory bank conflicts can negate optimization benefits**
   - Even with "better" global memory access patterns, bank conflicts can cause severe performance degradation

2. **Padding is a simple but effective solution**
   - Adding just one extra column can eliminate bank conflicts entirely

3. **Memory access patterns matter at multiple levels**
   - Global memory: Coalescing is crucial (e.g. loop reordering)
   - Shared memory: Bank conflicts must be avoided

4. **Always profile and measure**
   - Intuitive "optimizations" may actually harm performance
   - Use tools like NVIDIA Nsight to identify bottlenecks

## Building and Running

### Compilation
```bash
# Build and run C++ program
make run-cpp

# Build and run CUDA program
make run-cuda
```

## Dependencies

- CUDA Toolkit
- C++ compiler with OpenMP support
- `matrix_utils.h` and `matrix_utils.cpp` from `../matrix_addition/`

## Future Optimizations

Potential improvements for even better performance:
- Tensor Core utilization for supported GPUs
- Double buffering for overlapping computation and memory transfers
- Warp-level primitives for faster reductions
- Mixed precision computation where applicable
