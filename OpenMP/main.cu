#include <omp.h>
#include <stdio.h>
#include <iostream>

using namespace std;


// a simple kernel that simply increments each array element by b
__global__ void kernelAddConstant(int *g_a, const int b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_a[idx] += b;
}

// a predicate that checks whether each array elemen is set to its index plus b
int correctResult(int *data, const int n, const int b)
{
    for (int i = 0; i < n; i++)
        if (data[i] != i + b)
            return 0;

    return 1;
}

int main(int argc, char * argv[]){
	int gpu_count = 0;
	cudaGetDeviceCount(&gpu_count);

    unsigned int n = gpu_count* 8192 * 8192;
    cout << n << endl;
    unsigned int nbytes = n * sizeof(int);
    int *a = 0;     // pointer to data on the CPU
    int b = 3;      // value by which the array is incremented

    a = (int *)malloc(nbytes);

    if (0 == a)
    {
        printf("couldn't allocate CPU memory\n");
        return 1;
    }

    for (unsigned int i = 0; i < n; i++)
        a[i] = i;


    omp_set_num_threads(gpu_count);  // create as many CPU threads as there are CUDA devices
    //omp_set_num_threads(2*num_gpus);// create twice as many CPU threads as there are CUDA devices
    #pragma omp parallel
    {
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();

        // set and check the CUDA device for this CPU thread
        int gpu_id = -1;
        cudaSetDevice(cpu_thread_id % gpu_count);   // "% num_gpus" allows more CPU threads than GPU devices
        cudaGetDevice(&gpu_id);
        printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);

        int *d_a = 0;   // pointer to memory on the device associated with this CPU thread
        int *sub_a = a + cpu_thread_id * n / num_cpu_threads;   // pointer to this CPU thread's portion of data
        unsigned int nbytes_per_kernel = nbytes / num_cpu_threads;
        dim3 gpu_threads(128);  // 128 threads per block
        dim3 gpu_blocks(n / (gpu_threads.x * num_cpu_threads));

        cudaMalloc((void **)&d_a, nbytes_per_kernel);
        cudaMemset(d_a, 0, nbytes_per_kernel);
        cudaMemcpy(d_a, sub_a, nbytes_per_kernel, cudaMemcpyHostToDevice);
        kernelAddConstant<<<gpu_blocks, gpu_threads>>>(d_a, b);

        cudaMemcpy(sub_a, d_a, nbytes_per_kernel, cudaMemcpyDeviceToHost);
        cudaFree(d_a);

    }
    printf("---------------------------\n");


    bool bResult = correctResult(a, n, b);

    if (a) free(a); 

    cudaDeviceReset();

    exit(bResult ? EXIT_SUCCESS : EXIT_FAILURE);
}