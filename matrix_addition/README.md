# Matrix Addition

A simple CUDA project demonstrating matrix addition with both GPU (CUDA) and CPU implementations.

## Files

- `matrix_addition.cu` - CUDA implementation
- `matrix_addition.cpp` - CPU implementation  
- `matrix_utils.h/cpp` - Shared utility functions
- `Makefile` - Build configuration

## Quick Start

```bash
# Build both versions
make

# Build and run CUDA version
make run-cuda

# Build and run CPU version
make run-cpp

# Clean up executables
make clean
```

## Available Commands

| Command | Description |
|---------|-------------|
| `make` or `make all` | Build both CUDA and CPU versions |
| `make cuda` | Build CUDA version only |
| `make cpp` | Build CPU version only |
| `make run-cuda` | Build and run CUDA version |
| `make run-cpp` | Build and run CPU version |
| `make clean` | Remove executables |
| `make help` | Show all available commands |

## Requirements

- NVIDIA CUDA Toolkit
- GCC compiler
- CUDA-capable GPU (for CUDA version)

## Output

Both versions perform matrix addition on 1024×1024 matrices and display the first 5×5 portion of matrices A, B, and result C.
