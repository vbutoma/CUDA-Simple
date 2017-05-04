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
	}else {
		// if ((i < n) && (j < m)) c_result[i * m + j] = 255;
	}
}

extern __shared__ uchar sMins[];
extern __shared__ int posMins[];

__global__ void minimumKernel(uchar & output, uchar * res, const int patch_size, int n, int m){

	sMins[threadIdx.x] = (uchar)res[0];
	posMins[threadIdx.x] = 0;
	int xi, xj;
	for (xi = blockIdx.x; xi < n; xi += gridDim.x){
		int vectorBase = xi * n;
		int vectorEnd = vectorBase + n;
		xj = vectorBase + threadIdx.x;
		while ((xj < vectorEnd) && (xj < (n-1)*m)){
			xj += blockDim.x;
			register unsigned char d = res[xj];
			if (d < sMins[threadIdx.x]){
				sMins[threadIdx.x] = (uchar)d;
				posMins[threadIdx.x] = (int)xj;
			}
		}
	}

	__syncthreads();
	if (threadIdx.x == 0){
		register unsigned char min_value = sMins[0];
		register int min_pos = posMins[0];
		for (xj = 1; xj < blockDim.x; xj++){
			if (min_value > sMins[xj]){
				min_value = sMins[xj];
				min_pos = posMins[xj];
			}
		}
		if (min_value < output){
			output = min_value;
		}

	}

	__syncthreads();
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

			SAFE_CALL(cudaMemcpy(d_patch, patchMatrix, patchBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
			uchar ans1 = (uchar)255;
			correctionKernel<<<grid, block>>>(d_patch, d_sample, d_result, patch_size, N, M);
			//SAFE_CALL(cudaMemcpy())
			//minimumKernel<<<N, N, 512 * (sizeof(int) + sizeof(uchar))>>>(ans1, d_result, patch_size, N, M);
			//cout << (int)ans1 << " ";
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
