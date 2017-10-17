#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include"k.h"
#include<math.h>
#include </usr/local/cuda/include/cuda.h>
#define CMCPYHTD cudaMemcpyHostToDevice
#define CMCPYDTH cudaMemcpyDeviceToHost
#define BLOCK_WIDTH 16

extern  "C" K gpu_floydwarshall(K matrix);

/**Kernel for wake gpu
*
* @param reps dummy variable only to perform some action
*/
__global__ void wake_gpu_kernel(int reps) 
{
    I idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= reps) return;
}

/**Kernel for parallel Floyd Warshall algorithm on gpu
* 
* @param u number vertex of which is performed relaxation paths [v1, v2]
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param d matrix of shortest paths d(G)
*/
__global__ void fw_kernel(const unsigned int u, const unsigned int n, int * const d)
{
    I v1 = blockDim.y * blockIdx.y + threadIdx.y;
    I v2 = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (v1 < n && v2 < n) 
    {
        I newPath = d[v1 * n + u] + d[u * n + v2];
        I oldPath = d[v1 * n + v2];
        if (oldPath > newPath)
        {
            d[v1 * n + v2] = newPath;
        }
    }
}

K gpu_floydwarshall(K matrix)
{
    unsigned int V = sqrt(matrix->n);
    unsigned int n = V;
    // Alloc host data for G - graph, d - matrix of shortest paths
    unsigned int size = V * V;
    I *d = (int *) malloc (sizeof(int) * size);
    I *dev_d = 0;
    cudaStream_t cpyStream;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaSetDevice(0);

    // Initialize the grid and block dimensions here
    dim3 dimGrid((n - 1) / BLOCK_WIDTH + 1, (n - 1) / BLOCK_WIDTH + 1);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);

    // Create new stream to copy data
    cudaStreamCreate(&cpyStream);

    // Allocate GPU buffers for matrix of shortest paths d)
    cudaMalloc((void**)&dev_d, n * n * sizeof(int));
 
    // Wake up gpu
    wake_gpu_kernel<<<1, dimBlock>>>(32);

    // Copy input from host memory to GPU buffers.
    I *host_memoryd = (int*)&(kI(matrix)[0]);
    cudaMemcpyAsync(dev_d, host_memoryd, n * n * sizeof(int), CMCPYHTD, cpyStream);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    cudaDeviceSynchronize();

    // set preference for larger L1 cache and smaller shared memory
    cudaFuncSetCacheConfig(fw_kernel, cudaFuncCachePreferL1 );
    for (int u = 0; u <= (n-1); ++u)
    {
        fw_kernel<<<dimGrid, dimBlock>>>(u, n, dev_d);
    }

    // Check for any errors launching the kernel
    cudaGetLastError();

    // copy mem from gpu back to host
    cudaMemcpy(host_memoryd, dev_d, n * n * sizeof(int), CMCPYDTH);

    // free memory on gpu
    cudaFree(dev_d);

    // Delete allocated memory on host
    free(d);

    R r1(matrix);
}
