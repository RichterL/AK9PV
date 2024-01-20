#include <stdio.h>
#include <cuda.h>
#include <iostream>


#define N 1024
#define BLOCK_SIZE 256

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN( value ) {							\
    cudaError_t err = value;									\
    if( err != cudaSuccess ) {									\
        fprintf( stderr, "Error %s at line %d in file %s\n",	\
        cudaGetErrorString(err), __LINE__, __FILE__ );	        \
        exit( 1 );												\
    } }

// CUDA Kernel pro násobení matic
__global__ void matrixMultiply(int *a, int *b, int *c, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        int sum = 0;
        for (int i = 0; i < width; i++) {
            sum += a[row * width + i] * b[i * width + col];
        }
        c[row * width + col] = sum;
    }
}

int main() {
    int size = N * N * sizeof(int);
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    // cpu alloc
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // create A and B matrices
    for (int i = 0; i < N*N; i++) {
        a[i] = i + 1;
        b[i] = i + 1;
    }

    // allocate gpu memory
    CUDA_CHECK_RETURN( cudaMalloc((void **)&d_a, size) );
    CUDA_CHECK_RETURN( cudaMalloc((void **)&d_b, size) );
    CUDA_CHECK_RETURN( cudaMalloc((void **)&d_c, size) );

    // copy matrices
    CUDA_CHECK_RETURN( cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice) );
    CUDA_CHECK_RETURN( cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice) );

    CUDA_CHECK_RETURN( cudaDeviceSynchronize() );

    // run kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N / BLOCK_SIZE, N / BLOCK_SIZE);
    matrixMultiply<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);

    CUDA_CHECK_RETURN( cudaDeviceSynchronize() );


    // copy the result to CPU
    CUDA_CHECK_RETURN( cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost) );
    CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
    // free memory

    CUDA_CHECK_RETURN( cudaFree(d_a) );
    CUDA_CHECK_RETURN( cudaFree(d_b) );
    CUDA_CHECK_RETURN( cudaFree(d_c) );

    if (N == 3) {
        std::cout << "Matice A:" << std::endl;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << a[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << "Matice B:" << std::endl;
        free(a);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << b[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
        free(b);
        std::cout << std::endl;
        std::cout << "Matice C:" << std::endl;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << c[i * N + j] << " ";
            }
            std::cout << std::endl;
        }

        free(c);
    } else {
        free(a);free(b);free(c);
    }

    return 0;
}