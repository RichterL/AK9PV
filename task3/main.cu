#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <png++/png.hpp>
#include <png++/image.hpp>
#include "pngio.h"

#define BLOCK_SIZE (16u)

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

__global__ void blurKernel(const unsigned char *inputImage, unsigned char *outputImage, int width, int height, int kernelSize = 3) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int offset = y * width * 3 + x * 3;

        int rSum = 0;
        int gSum = 0;
        int bSum = 0;
        int count = 0;

        // Průchod okolními pixely v kernelu
        for (int ky = -kernelSize; ky <= kernelSize; ky++) {
            for (int kx = -kernelSize; kx <= kernelSize; kx++) {
                int px = x + kx;
                int py = y + ky;

                if (px >= 0 && px < width && py >= 0 && py < height) {
                    int pOffset = py * width * 3 + px * 3;
                    rSum += inputImage[pOffset];
                    gSum += inputImage[pOffset + 1];
                    bSum += inputImage[pOffset + 2];
                    count++;
                }
            }
        }

        // Výpočet průměrné hodnoty pro každý kanál
        outputImage[offset] = static_cast<unsigned char>(rSum / count);
        outputImage[offset + 1] = static_cast<unsigned char>(gSum / count);
        outputImage[offset + 2] = static_cast<unsigned char>(bSum / count);
    }
}

__global__ void noiseKernel(const unsigned char *inputImage, unsigned char *outputImage, int width, int height, float noiseIntensity = 0.8) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int offset = y * width * 3 + x * 3;

        curandState state;
        curand_init(0, offset, 0, &state);  // Inicializace generátoru náhodných čísel

        float noiseR = curand_uniform(&state) * noiseIntensity;
        float noiseG = curand_uniform(&state) * noiseIntensity;
        float noiseB = curand_uniform(&state) * noiseIntensity;

        outputImage[offset] = min(255, max(0, inputImage[offset] + static_cast<unsigned char>(noiseR * 255)));
        outputImage[offset + 1] = min(255, max(0, inputImage[offset + 1] + static_cast<unsigned char>(noiseG * 255)));
        outputImage[offset + 2] = min(255, max(0, inputImage[offset + 2] + static_cast<unsigned char>(noiseB * 255)));
    }
}

__global__ void bwKernel(const unsigned char *inputImage, unsigned char *outputImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int offset = y * width * 3 + x * 3;

        unsigned char r = inputImage[offset];
        unsigned char g = inputImage[offset + 1];
        unsigned char b = inputImage[offset + 2];

        // Výpočet průměrné hodnoty kanálů R, G a B
        unsigned char grayValue = static_cast<unsigned char>((r + g + b) / 3);

        outputImage[offset] = grayValue;
        outputImage[offset + 1] = grayValue;
        outputImage[offset + 2] = grayValue;
    }
}

__global__ void edgeDetectionKernel(const unsigned char *inputImage, unsigned char *outputImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int offset = y * width * 3 + x * 3;

        // Sobel operátory pro detekci hran
        int sobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int sobelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

        int rGradientX = 0;
        int gGradientX = 0;
        int bGradientX = 0;

        int rGradientY = 0;
        int gGradientY = 0;
        int bGradientY = 0;

        // Průchod okolními pixely v kernelu a aplikace Sobel operátoru
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int px = x + kx;
                int py = y + ky;

                if (px >= 0 && px < width && py >= 0 && py < height) {
                    int pOffset = py * width * 3 + px * 3;
                    int kernelValueX = sobelX[ky + 1][kx + 1];
                    int kernelValueY = sobelY[ky + 1][kx + 1];

                    rGradientX += inputImage[pOffset] * kernelValueX;
                    gGradientX += inputImage[pOffset + 1] * kernelValueX;
                    bGradientX += inputImage[pOffset + 2] * kernelValueX;

                    rGradientY += inputImage[pOffset] * kernelValueY;
                    gGradientY += inputImage[pOffset + 1] * kernelValueY;
                    bGradientY += inputImage[pOffset + 2] * kernelValueY;
                }
            }
        }

        // Celkový gradient pro každý kanál
        int rGradient = abs(rGradientX) + abs(rGradientY);
        int gGradient = abs(gGradientX) + abs(gGradientY);
        int bGradient = abs(bGradientX) + abs(bGradientY);

        // Výstupní hodnoty pixelu reprezentují detekci hran
        outputImage[offset] = static_cast<unsigned char>(min(255, max(0, rGradient)));
        outputImage[offset + 1] = static_cast<unsigned char>(min(255, max(0, gGradient)));
        outputImage[offset + 2] = static_cast<unsigned char>(min(255, max(0, bGradient)));
    }
}



int main()
{
    // Vytvoření seznamu možností
    std::vector<std::string> options = {"Blur", "Noise", "Black & White", "Detect edges"};

    std::cout << "Vyberte kernel:" << std::endl;

    // Vypsání možností
    for (size_t i = 0; i < options.size(); ++i) {
        std::cout << i + 1 << ". " << options[i] << std::endl;
    }

    int choice;
    std::cout << "Choose an operation to be performed on input.png: ";
    std::cin >> choice;

    if (choice >= 1 && choice <= options.size()) {
        std::cout << "You chose ";
        switch (choice) {
            case 1:
                std::cout << "Blur" << std::endl;
            break;
            case 2:
                std::cout << "Noise" << std::endl;
            break;
            case 3:
                std::cout << "Black & White" << std::endl;
            break;
            case 4:
                std::cout << "Detect edges" << std::endl;
            break;
        }
    } else {
        std::cerr << "Not a valid choice" << std::endl;
        return 1;
    }

    // load the image
    png::image<png::rgb_pixel> inputImage("../input.png");
    unsigned int width = inputImage.get_width();
    unsigned int height = inputImage.get_height();

    int size = width * 3 * height * sizeof(unsigned char);
    unsigned char *h_inputImage = new unsigned char[ size ];
    unsigned char *h_outputImage = new unsigned char[ size ];

    pvg::pngToRgb( h_inputImage, inputImage );

    // allocate gpu memory
    unsigned char *d_inputImage, *d_outputImage;
    CUDA_CHECK_RETURN( cudaMalloc(&d_inputImage, size) );
    CUDA_CHECK_RETURN( cudaMalloc(&d_outputImage, size) );

    std::cout << "cuda malloc OK" << std::endl;

    // copy image to gpu
    CUDA_CHECK_RETURN( cudaMemcpy( d_inputImage, h_inputImage, size, cudaMemcpyHostToDevice) );

    std::cout << "png copy OK" << std::endl;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    std::cout << "kernel starting" << std::endl;


    std::string outputFilename;
    // choose and launch kernel
    switch (choice) {
        case 1:
            blurKernel<<<grid, block>>>(d_inputImage, d_outputImage, width, height);
            outputFilename = "../blur.png";
        break;
        case 2:
            noiseKernel<<<grid, block>>>(d_inputImage, d_outputImage, width, height);
            outputFilename = "../noise.png";
        break;
        case 3:
            bwKernel<<<grid, block>>>(d_inputImage, d_outputImage, width, height);
            outputFilename = "../bw.png";
        break;
        case 4:
            edgeDetectionKernel<<<grid, block>>>(d_inputImage, d_outputImage, width, height);
            outputFilename = "../edge.png";
        break;
    }


    std::cout << "kernel OK" << std::endl;

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }
    CUDA_CHECK_RETURN( cudaDeviceSynchronize() );


    // copy image back to CPU
    CUDA_CHECK_RETURN( cudaMemcpy( h_outputImage, d_outputImage, size, cudaMemcpyDeviceToHost ) );

    // save the image
    pvg::rgbToPng( inputImage, h_outputImage );
    inputImage.write(outputFilename);

    // free the allocated memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return 0;
}
