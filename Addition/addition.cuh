#ifndef ADDITION_CUH_
#define BILATERAL_FILTER_CUH_

#include<iostream>
#include<cstdio>
#include<cuda_runtime.h>

using std::cout;
using std::endl;

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

__global__ void additionKernel(unsigned int * a,
															 unsigned int * b,
														   unsigned int * c,
														 	 const int n,
														 	 const int m)
{
	//int outputXIndex = blockIdx.x * blockDim.x + threadIdx.x;
	//int outputYIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < n && j < m){
		int my_ij = i*m + j;
		c[my_ij] = a[my_ij] + b[my_ij];
	}
}

void additionCPU(int * A, int * B, int * C, int N, int M){
	GpuTimer timer;
	timer.Start();
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			*(C + i*M + j) = *(A + i*M + j) + *(B + i*M + j);

	timer.Stop();
	printf("CPU code ran in: %f msecs.\n", timer.Elapsed());
}

void additionGPU(int *A, int * B, int * D, int N, int M){


	const int aBytes = N * M * sizeof(int);
	const int bBytes = N * M * sizeof(int);
	const int dBytes = N * M * sizeof(int);

	unsigned int *d_a, *d_b, *d_d;
	cudaMalloc((void**)&d_a, aBytes);
	cudaMalloc((void**)&d_b, bBytes);
	cudaMalloc((void**)&d_d, dBytes);

	SAFE_CALL(cudaMemcpy(d_a, A, aBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_b, B, bBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_d, D, dBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");


	GpuTimer timer;
	timer.Start();
	const dim3 block(512, 512);
	dim3 grid(N / block.x + 1, M / block.y + 1);
	additionKernel<<<grid, block>>>(d_a, d_b, d_d, N, M);
	timer.Stop();
	printf("GPU code ran in: %f msecs.\n", timer.Elapsed());
	//SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");
	cout <<"hui"<<endl;

	SAFE_CALL(cudaMemcpy(D, d_d, dBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

	SAFE_CALL(cudaFree(d_a),"CUDA Free Failed");
	SAFE_CALL(cudaFree(d_b),"CUDA Free Failed");
	SAFE_CALL(cudaFree(d_d),"CUDA Free Failed");
}

#endif /* ADDITION_CUH_ */
