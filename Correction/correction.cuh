#ifndef CORRECTION_CUH_

#include<iostream>
#include<cstdio>
#include<cuda_runtime.h>

using namespace std;

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

__device__ int cuda_abs(const int x){
	return (x >= 0 ? x : -x);
}

__device__ int cuda_max(int x, int y){
	return (x > y ? x : y);
}

__global__ void correctionKernel(uchar * a_patch,
															 uchar * b_sample,
															 uchar * c_result,
															 const int patch_size,
														 	 const int n,
														 	 const int m)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;

	int temp = 0, xi, xj;
	if ((i >= patch_size) && (i < n - patch_size-1) && (j < m - patch_size-1) && (j >= patch_size)){

		for (xi = 0; xi < (2*patch_size+1); xi++){
			for (xj = 0; xj < (2 * patch_size + 1); xj++){
				int patch_xy = xi * patch_size + xj;
				int si = xi - patch_size + i, sj = xj - patch_size + j;
				int sample_xy = si * m + sj;
				temp = cuda_max(cuda_abs(a_patch[patch_xy] - b_sample[sample_xy]), temp);
			}
		}
		c_result[i * m + j] = (uchar)temp;
	}
}


__global__ void minimumKernel(uchar * res, const int patch_size,
const int n, const int m){
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i >= patch_size) && (i < n - patch_size-1) && (j < m - patch_size-1) && (j >= patch_size)){
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

	uchar *d_patch, *d_sample, *d_result;
	const int patchBytes = (patch_size * 2 + 1) * (patch_size * 2 + 1) + sizeof(uchar);
	const int sampleBytes = N * M * sizeof(uchar);
	const int resultBytes = N * M * sizeof(uchar);
	SAFE_CALL(cudaMalloc((void**)&d_patch, patchBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc((void**)&d_sample, sampleBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc((void**)&d_result, resultBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMemcpy(d_sample, sampleMatrix, sampleBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

	int patchn = (patch_size * 2 + 1);
	int patchm = patchn;

	uchar ** patchMatrix = new uchar*[patchn];
	for (int i = 0; i < image.rows; i++){
		patchMatrix[i] = new uchar[patchm];
  }
	cout << image.rows << " " << image.cols << endl;
	int ans;
	for (int i = border; i < image.rows - border; i++){
		for (int j = border; j < image.cols - border; j++){
			cv::Point patch_tl(std::max(j - patch_size, 0), std::max(i - patch_size, 0));
      cv::Point patch_br(std::min(j + patch_size + 1, image.cols), std::min(i + patch_size + 1, image.rows));
      cv::Mat tex_patch = image(cv::Rect(patch_tl, patch_br));
			//cout << i << " " << j << " " << tex_patch.rows << " " << tex_patch.cols << endl;
			for (int k = 0; k < tex_patch.rows; k++){
				uchar * pixel = tex_patch.ptr<uchar>(i);
				for (int l = 0; l < tex_patch.cols; l++)
					patchMatrix[k][l] = (uchar)*pixel++;
			}
			ans = 1000; // infinity
			SAFE_CALL(cudaMemcpy(d_patch, patchMatrix, patchBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
			correctionKernel<<<grid, block>>>(d_patch, d_sample, d_result, patch_size, N, M);
			minimumKernel<<<grid, block>>>(d_result, patch_size, N, M);
		}
	}

	timer.Stop();
	printf("GPU code ran in: %f msecs.\n", timer.Elapsed());
	for (int i = 0; i < sample.rows; i++){
		delete [] imageMatrix[i];
  }

	delete [] imageMatrix;

	for (int i = 0; i < sample.rows; i++){
		delete [] sampleMatrix[i];
  }
	delete [] sampleMatrix;



	SAFE_CALL(cudaFree(d_result),"CUDA Free Failed");
	SAFE_CALL(cudaFree(d_sample),"CUDA Free Failed");
	SAFE_CALL(cudaFree(d_patch),"CUDA Free Failed");
	cout << "EXIT" << endl;
}

#endif /* CORRECTION_CUH_ */
