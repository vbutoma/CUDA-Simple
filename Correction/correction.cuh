#ifndef CORRECTION_CUH_

#include<iostream>
#include<cstdio>
#include<cuda_runtime.h>

using std::cout;
using std::endl;

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if(err!=cudaSuccess)
	{
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

__global__ void correctionKernel(uchar * a_patch,
															 uchar * b_sample,
															 const int patch_size,
														 	 const int n,
														 	 const int m)
{
	//int outputXIndex = blockIdx.x * blockDim.x + threadIdx.x;
	//int outputYIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < n && j < m){
		int my_ij = i*m + j;
	}
}

void correctionGPU(cv::Mat sample, cv::Mat image, int patch_size, int border){
	// GRAY SCALE IMAGES
	assert(patch_size == border);
	uchar ** sampleMatrix = new uchar*[sample.rows];
	for (int i = 0; i < sample.rows; i++){
		sampleMatrix[i] = new uchar[sample.cols];
		uchar* pixel = sample.ptr<uchar>(i);
    for(int j = 0; j < sample.cols; j++){
			sampleMatrix[i][j] = (int)*pixel++;
    }
  }
	uchar ** imageMatrix = new uchar*[image.rows];
	for (int i = 0; i < image.rows; i++){
		imageMatrix[i] = new uchar[image.cols];
		uchar* pixel = image.ptr<uchar>(i);
    for(int j = 0; j < image.cols; j++){
			imageMatrix[i][j] = (int)*pixel++;
    }
  }

	GpuTimer timer;
	timer.Start();
	int N = sample.rows, M = sample.cols;
	const dim3 block(512, 512);
	dim3 grid(N / block.x + 1, M / block.y + 1);

	uchar *d_patch, *d_sample;
	const int patchBytes = (patch_size * 2 + 1) * (patch_size * 2 + 1) + sizeof(uchar);
	const int sampleBytes = N * M * sizeof(uchar);
	cudaMalloc((void**)&d_patch, patchBytes);
	cudaMalloc((void**)&d_sample, sampleBytes);

	for (int i = border; i < image.rows - border; i++){
		for (int j = border; j < image.cols - border; j++){
			correctionKernel<<<grid, block>>>(d_patch, d_sample, patch_size, N, M);
		}
	}

	timer.Stop();
	printf("GPU code ran in: %f msecs.\n", timer.Elapsed());

	SAFE_CALL(cudaFree(d_patch),"CUDA Free Failed");
	SAFE_CALL(cudaFree(d_sample),"CUDA Free Failed");

}

#endif /* CORRECTION_CUH_ */
