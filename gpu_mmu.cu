#include </usr/local/cuda/include/cuda.h>
#include"stdio.h"
#include"k.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>
#include <time.h>
#include <sys/time.h>
#define uS_PER_SEC 1000000
#define uS_PER_mS 1000
#define N  1000
#define M 1000

// Export the function we will load into kdb+
extern  "C" K gpu_mmf(K A, K rA, K cA, K B, K rB, K cB, K C);

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
// m= nr_rows_A
// k= nr_cols_A
// n= nr_cols_B
void gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n) {
    int lda=k,ldb=n,ldc=m;
    timeval t1, t2;
    const double alf = 1;
    const double bet = 0;
    const double *alpha = &alf;
    const double *beta = &bet;

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // start timer
    gettimeofday(&t1, NULL);

    // Do the actual multiplication
    // CUBLAS_OP_T means input is row major, CUBLAS_OP_N means input is column major
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    gettimeofday(&t2, NULL);
    float et2 = (((t2.tv_sec*uS_PER_SEC)+t2.tv_usec) - ((t1.tv_sec*uS_PER_SEC)+t1.tv_usec))/(float)uS_PER_mS;
    printf("GPU time = %fms\n", et2);
  
    // Destroy the handle
    cublasDestroy(handle);
}

K gpu_mmf(K A, K rA, K cA, K B, K rB, K cB,  K C) {
    // Allocate 3 arrays on CPU
    int nr_rows_A = rA->n;
    int nr_cols_A = cA->n;
    int nr_rows_B = rB->n;
    int nr_cols_B = cB->n;
    int nr_rows_C = nr_rows_A;
    int nr_cols_C = nr_cols_B;

    // allocate memory, host arrays
    double *h_A = (double *)malloc(nr_rows_A * nr_cols_A * sizeof(double));
    double *h_B = (double *)malloc(nr_rows_B * nr_cols_B * sizeof(double));
    double *h_C = (double *)malloc(nr_rows_C * nr_cols_C * sizeof(double));

    // Allocate 3 arrays on GPU, device arrays
    double *d_A, *d_B, *d_C;
    double *host_memoryA = (double*) &(kF(A)[0]);
    double *host_memoryB = (double*) &(kF(B)[0]);
    double *host_memoryC = (double*) &(kF(C)[0]);
    size_t sizeA = nr_rows_A * nr_cols_A * sizeof(double);
    size_t sizeB = nr_rows_B * nr_cols_B * sizeof(double);
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(double));

    // If you already have useful values in A and B you can copy them in GPU:
    cudaMemcpy(d_A, host_memoryA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, host_memoryB, sizeB, cudaMemcpyHostToDevice);

    // Multiply A and B on GPU
    gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);

    // Copy the result on host memory
    cudaMemcpy(host_memoryC,d_C,nr_rows_C * nr_cols_C * sizeof(double),cudaMemcpyDeviceToHost);

    //Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);    

    // Free CPU memory
    free(h_A);
    free(h_B);
    free(h_C);

    R r1(C);
}
