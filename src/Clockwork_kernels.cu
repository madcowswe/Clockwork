// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "device_functions.h"

const int N = 7;
const int blocksize = 7;

__global__
	void hello(char *a, int *b)
{
	a[threadIdx.x] += b[threadIdx.x];
}


int testcuda()
{
	char a[N] = "Hello ";
	int b[N] = {15, 10, 6, 0, -11, 1, 0};

	char *ad;
	int *bd;
	const int csize = N*sizeof(char);
	const int isize = N*sizeof(int);

	printf("%s", a);

	cudaMalloc( (void**)&ad, csize );
	cudaMalloc( (void**)&bd, isize );
	cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice );
	cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice );

	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );
	hello<<<dimGrid, dimBlock>>>(ad, bd);
	cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost );
	cudaFree( ad );

	printf("%s\n", a);
	return 0;
}

template <unsigned N>
__global__ void Clockwork(uint32_t* staticbank,
						  uint32_t* regbank,
						  uint32_t* sharedbank1,
						  uint32_t* sharedbank2,
						  //uint32_t N,
						  int* bestiBuff,
						  int* bestiBuffHead)
{
	unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;

	uint32_t xstatic = staticbank[threadID];

	__shared__ uint32_t sb1[N];
	__shared__ uint32_t sb2[N];

	for (int i = threadIdx.x; i < N; i += blockDim.x)
	{
		sb1[i] = sharedbank1[i];
		sb2[i] = sharedbank2[i];
	}

	for (int rbidx = 0; rbidx < N; rbidx += 8)
	{
		uint32_t rb[8];
		for (int i = 0; i < 8; i++)
		{
			rb[i] = regbank[rbidx + i];
		}

		for (int i = 0; i < N; i++)
		{
			uint32_t acc1 = xstatic ^ sb2[i];
			for (int j = 0; j < N; j++)
			{
				uint32_t acc2 = acc1 ^ sb1[i];
				for (int k = 0; k < 8; k++)
				{
					//Only bother cheking for perfect xor, same as equal
					if (acc2 == rb[i])
					{
						int storeloc = atomicAdd(bestiBuffHead, 4);
						bestiBuff[storeloc] = j;
						bestiBuff[storeloc+1] = i;
						bestiBuff[storeloc+2] = 8*rbidx + k;
						bestiBuff[storeloc+3] = threadID;
					}
				}
			}
		}
	}

}