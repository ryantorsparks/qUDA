# cudaq

**Background**

I've written some matrix related functions into cuda/kdb capi code, mainly for performance. I was originally motivated by trying to find faster matrix multiply, for neural network/convolutional neural network training (which I plan to talk about at a future kx meetup, hopefully this year). Nick Psaris gave me the idea of trying to integrate cuBLAS in his kx con machine learning presentation.

The functions are:
* floydwarshall: the standard shortest past algorithm for shortest paths between all nodes
* credit matrix: a different take on shortest paths, given a credit matrix (the max credit that a counterparty can trade with another), what is the max possible credit between all counterparties going via alternate paths)
* gpu_mmu: a cuBLAS version of matrix multiply

**GPU hardware**

All of my functions were written and tested using a Tesla K80 gpu

**Performance**

The performance of these 3 are all significantly faster than the fastest kdb code I've seen.Floyd warshall and credit matrix functions seem to go about 10 times faster than the best q code I've seen, which is (a slightly slimmer version of http://code.kx.com/q/cookbook/shortestpath/):
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

Here's a more detailed comparison of the floyd warshall algo, running on various versions. Note that my gpu server only has one cpu, hence I can't run any slaves on it, and my sever with multiple cpus has no gpu on it

| function   | 2000x2000 gpu server 0 slaves | 2000x2000 on fast box+6 slaves | 4000x4000 gpu server 0 slaves | 4000x4000 fast box+6 slaves | 
|------------|-------------------------------|--------------------------------|-------------------------------|-----------------------------| 
| bridge     | wsfull                        | 46274 32833633920              | wsfull                        | didn't try                  | 
| bridge1    | 17568 65650480                | 6225 32833952                  | didnt' try                    | 51735 131203488             | 
| bridge2    | 13365 49250400                | 3446 32826032                  | 134495 196802624              | 33971 131187376             | 
| bridgec    | 8202 592                      | 7341 592                       | 75727 592                     | 106828 592                  | 
| bridgecuda | 388 592                       | n/a                            | 2890 592                      | n/a                         | 

where the functions are
```
bridge:k){x&&/''x+/:\:+x}
bridge1:k){x&(min'(+x)+\:)':x}
bridge2:k){x&.Q.fc[{(min y+)'x}[+x]';x]}
bridgec: c version of cuda func (.so object loaded into kdb)
bridgecuda:cuda func
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
// read in cuda func using 2:
q)creditcuda:`creditmatrix 2:(`gpu_creditmatrix;1)

// q func
q)creditq:{x&.Q.fc[{(min y+)each x}[flip x]each;x]}

// 4 million element input matrix (cuda expects flat version)
q)m2:2000 2000#m:40000000?100i

// q version (I had no slaves available on my gpu server, so this should be faster)
q)\t creditq/[m2]
36207

// cuda version
q)\t res2:creditcuda m
456

q) res2~raze res
1b
```
