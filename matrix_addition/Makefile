# Makefile for Matrix Addition
NVCC = nvcc
GCC = g++

# Build flags
NVCC_FLAGS = -O3
GCC_FLAGS = -O3 -std=c++11

# Source files
CUDA_SRC = matrix_addition.cu
CPP_SRC = matrix_addition.cpp
UTILS_SRC = matrix_utils.cpp
UTILS_H = matrix_utils.h

# Executables
CUDA_EXEC = matrix_addition_cuda
CPP_EXEC = matrix_addition_cpp

# Default target
all: cuda cpp

# Build targets
cuda: $(CUDA_EXEC)
cpp: $(CPP_EXEC)

$(CUDA_EXEC): $(CUDA_SRC) $(UTILS_SRC) $(UTILS_H)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(CUDA_SRC) $(UTILS_SRC)

$(CPP_EXEC): $(CPP_SRC) $(UTILS_SRC) $(UTILS_H)
	$(GCC) $(GCC_FLAGS) -o $@ $(CPP_SRC) $(UTILS_SRC)

# Run targets
run-cuda: $(CUDA_EXEC)
	./$(CUDA_EXEC)

run-cpp: $(CPP_EXEC)
	./$(CPP_EXEC)

# Cleanup
clean:
	rm -f $(CUDA_EXEC) $(CPP_EXEC)

# Help
help:
	@echo "Available targets:"
	@echo "  all       - Build both versions"
	@echo "  cuda      - Build CUDA version"
	@echo "  cpp       - Build CPU version"
	@echo "  run-cuda  - Build and run CUDA version"
	@echo "  run-cpp   - Build and run CPU version"
	@echo "  clean     - Remove executables"
	@echo "  help      - Show this help"

.PHONY: all cuda cpp run-cuda run-cpp clean help