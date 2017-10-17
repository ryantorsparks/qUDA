# cudaq

**Background**

I've written some matrix related functions into cuda/kdb capi code, mainly for performance. I was originally motivated by trying to find faster matrix multiply, for neural network/convolutional neural network training (which I plan to talk about at a future kx meetup, hopefully this year). Nick Psaris gave me the idea of trying to integrate cuBLAS in his kx con machine learning presentation.

The functions are:
* floydwarshall: the standard shortest past algorithm for shortest paths between all nodes
* credit matrix: a different take on shortest paths, given a credit matrix (the max credit that a counterparty can trade with another), what is the max possible credit between all counterparties going via alternate paths)
* gpu_mmu: a cuBLAS version of matrix multiply

**Performance**

The performance of these 3 are all significantly faster than the fastest kdb code I've seen.Floyd warshall and credit matrix functions seem to go about 10 times faster than the best q code I've seen, which is:
```
// floyd warshall k and q equivalent, benefits from slaves, credit matrix is very similar   
k){x&.Q.fc[{(min y+)'x}[+x]';x]}
q){x&.Q.fc[{(min y+)each x}[flip x]each;x]}
```
The matrix multiplication code is significantly faster too, 10 times faster than even qml:

```
// cuda BLAS requires flat input matrices, so have aflat and bflat for that input
// for my use cases, this is actually a good thing as I often need to flatten/reshape before/after
// 2 matrices, a (1000x2000) and b(2000x3000) 

q) a:1000 2000#aflat:2000000?10f
q) b:2000 3000#bflat:6000000?10f
q) empty:3000000#0f  // init the empty object to fill result in, takes about 10 millis if you need to do it each time
q)\t flatres:.gpu.mm[aflat;1000;2000;bflat;2000;3000;empty]
time to allocate host and device array mems: 1.358000ms
time to copy inputs to GPU: 9.041000ms
time to perform cublas matrix multiply: 0.024000ms
time to copy result from GPU back to host: 3.494000ms
40
```

As  you can see, the actual matrix multiply was almost immeasurable (0.00002), but it spent about 15 milis transporting the data to and from the gpu, and I guess a a bit of time preparing things around it.

As a comparison, here's the latest 3.5 native mmu, and qml's mm

```
q)\t res2:mmu[a;b]
2051

/ using qml
q)\t res3:.qml.mm[a;b]
485

// cublas returns column-major matrix, so if you want to compare, need to reshape/invert 
// (I have a c func for this too which helps)
q)res2~flip 3000 1000#flatres 
1b
```

**Compiling**

To compile these, I've just been using:
```
$ cat makeqcuda.sh
# e.g. $./makeqcuda floydwarshall
nvcc --compiler-options '-fPIC -DKXVER=3 -O2' -o $QHOME/l64/gpu_mm.so --shared -lcurand -lcublas gpu_mm.cu
```

**Loading**

Load into q like a c object:
```
// this example has no slaves, so the pefromance of the q func could be faster
q)creditcuda:`creditmatrix 2:(`gpu_creditmatrix;1)
q)creditq:{x&.Q.fc[{(min y+)each x}[flip x]each;x]}
q)m2:2000 2000#m:40000000?100i
12 10 1 90 73 90 43 90 84 63 93 54 38 97 88 58 68 45 2 39 64 49 82 40 88 77 3..
q)\t res:creditcuda m
14834
q)\t 0N!raze[res]~creditcuda m
1b
456
```
