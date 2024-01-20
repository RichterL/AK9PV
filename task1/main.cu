#include <stdio.h>
#include <cuda.h>
#include <iostream>

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN( value ) {							\
cudaError_t err = value;									\
if( err != cudaSuccess ) {									\
fprintf( stderr, "Error %s at line %d in file %s\n",	\
cudaGetErrorString(err), __LINE__, __FILE__ );	\
exit( 1 );												\
} }

#define VECT_SIZE (256u)
#define BLOCK_SIZE (256u)

__global__ void vectorFill(int *data, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < n) data[i] = i + 1;
}

__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1000 * 1000 * 1000;
    int *d_a, *d_b, *d_c;
    constexpr size_t size = n * sizeof(int);
    int *c = (int *) malloc(size);

    int blockSize = BLOCK_SIZE;
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CHECK_RETURN( cudaMalloc((void **)&d_a, size) );
    CUDA_CHECK_RETURN( cudaMalloc((void **)&d_b, size) );
    CUDA_CHECK_RETURN( cudaMalloc((void **)&d_c, size) );

    vectorFill<<<gridSize, blockSize>>>(d_a, n);
    vectorFill<<<gridSize, blockSize>>>(d_b, n);
    vectorFill<<<gridSize, blockSize>>>(d_c, n);
    CUDA_CHECK_RETURN( cudaDeviceSynchronize() );


    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_RETURN( cudaDeviceSynchronize() );


    CUDA_CHECK_RETURN( cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost) );

    free(c);
    CUDA_CHECK_RETURN(cudaFree(d_a) );
    CUDA_CHECK_RETURN(cudaFree(d_b) );
    CUDA_CHECK_RETURN(cudaFree(d_c) );


    return 0;
}
