// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <vector>
#include <thrust/device_vector.h>
#include <thrust/transform.h>



typedef std::pair<std::pair<uint64_t, uint64_t>, std::pair<uint64_t, uint64_t>>
	wide_as_pair;
typedef std::pair<wide_as_pair, std::vector<uint32_t>>
	point_idx_pair;
typedef thrust::pair<thrust::pair<uint64_t, uint64_t>, thrust::pair<uint64_t, uint64_t>>
	wide_as_pair_GPU;

struct bigint_t
{
	uint32_t limbs[8];
};

/*
	((uint64_t)roundInfo->c[3] << 32) + roundInfo->c[2],
	((uint64_t)roundInfo->c[1] << 32) + roundInfo->c[0],
std::make_pair(
	std::make_pair(
	((uint64_t)point_preload.limbs[7] << 32) + point_preload.limbs[6], 
	((uint64_t)point_preload.limbs[5] << 32) + point_preload.limbs[4]
),
	std::make_pair(
	((uint64_t)point_preload.limbs[3] << 32) + point_preload.limbs[2], 
	((uint64_t)point_preload.limbs[1] << 32) + point_preload.limbs[0]
	*/

/*! Add together two n-limb numbers, returning the carry limb.
	\note the output can also be one of the inputs
*/
__device__
uint32_t wide_add_GPU(unsigned n, uint32_t *res, const uint32_t *a, const uint32_t *b)
{
	uint64_t carry=0;
	for(unsigned i=0;i<n;i++){
		uint64_t tmp=uint64_t(a[i])+b[i]+carry;
		res[i]=uint32_t(tmp&0xFFFFFFFFULL);
		carry=tmp>>32;
	}
	return carry;
}

/*! Add a single limb to an n-limb number, returning the carry limb
	\note the output can also be the input
*/
__device__
uint32_t wide_add_GPU(unsigned n, uint32_t *res, const uint32_t *a, uint32_t b)
{
	uint64_t carry=b;
	for(unsigned i=0;i<n;i++){
		uint64_t tmp=a[i]+carry;
		res[i]=uint32_t(tmp&0xFFFFFFFFULL);
		carry=tmp>>32;
	}
	return carry;
}

/*! Multiply two n-limb numbers to produce a 2n-limb result
	\note All the integers must be distinct, the output cannot overlap the input */
__device__
void wide_mul_GPU(unsigned n, uint32_t *res_hi, uint32_t *res_lo, const uint32_t *a, const uint32_t *b)
{
	//assert(res_hi!=a && res_hi!=b);
	//assert(res_lo!=a && res_lo!=b);
	
	uint64_t carry=0, acc=0;
	for(unsigned i=0; i<n; i++){
		for(unsigned j=0; j<=i; j++){
			//assert( (j+(i-j))==i );
			uint64_t tmp=uint64_t(a[j])*b[i-j];
			acc+=tmp;
			if(acc < tmp)
				carry++;
			//fprintf(stderr, " (%d,%d)", j,i-j);
		}
		res_lo[i]=uint32_t(acc&0xFFFFFFFFull);
		//fprintf(stderr, "\n  %d : %u\n", i, res_lo[i]);
		acc= (carry<<32) | (acc>>32);
		carry=carry>>32;
	}
	
	for(unsigned i=1; i<n; i++){
		for(unsigned j=i; j<n; j++){
			uint64_t tmp=uint64_t(a[j])*b[n-j+i-1];
			acc+=tmp;
			if(acc < tmp)
				carry++;
			//fprintf(stderr, " (%d,%d)", j,n-j+i-1);
			//assert( (j+(n-j))==n+i );
		}
		res_hi[i-1]=uint32_t(acc&0xFFFFFFFFull);
		//fprintf(stderr, "\n  %d : %u\n", i+n-1, res_hi[i-1]);
		acc= (carry<<32) | (acc>>32);
		carry=carry>>32;
	}
	res_hi[n-1]=acc;
}

struct genpoint
{
	uint32_t point_preload[8];
	const unsigned numsteps;
	uint32_t c[4];

	genpoint(const uint32_t* const ppre_in, unsigned ns_in, const uint32_t* const c_in) :
		numsteps(ns_in)
	{
		for (int i = 0; i < 8; i++)
		{
			point_preload[i] = ppre_in[i];
		}

		for (int i = 0; i < 4; i++)
		{
			c[i] = c_in[i];
		}
	}

	__device__
	wide_as_pair_GPU operator()(uint32_t idx){

		uint32_t point[8];
		point[0] = idx;
		for (int i = 1; i < 8; i++)
		{
			point[i] = point_preload[i];
		}

		for (int i = 0; i < numsteps; i++)
		{
			bigint_t tmp;
			// tmp=lo(x)*c;
			wide_mul_GPU(4, tmp.limbs+4, tmp.limbs, point, c);
			// [carry,lo(x)] = lo(tmp)+hi(x)
			uint32_t carry=wide_add_GPU(4, point, tmp.limbs, point+4);
			// hi(x) = hi(tmp) + carry
			wide_add_GPU(4, point+4, tmp.limbs+4, carry);

			// overall:  tmp=lo(x)*c; x=tmp+hi(x)
		}

		return thrust::make_pair(
			thrust::make_pair(
			((uint64_t)point[7] << 32) + point[6], 
			((uint64_t)point[5] << 32) + point[4]
		),
			thrust::make_pair(
			((uint64_t)point[3] << 32) + point[2], 
			((uint64_t)point[1] << 32) + point[0]
		));
	}
};

namespace bitecoin{

	//TODO: zip into vector of pairs in place
	std::vector<wide_as_pair> genpoints_on_GPU (
		unsigned hashsteps,
		const uint32_t* const c,
		const uint32_t* const point_preload,
		const std::vector<uint32_t> &indexbank
	){

		thrust::device_vector<uint32_t> indexbank_GPU = indexbank;
		thrust::device_vector<wide_as_pair_GPU> output_GPU(indexbank.size());

		thrust::transform(indexbank_GPU.begin(), indexbank_GPU.end(), output_GPU.begin(), genpoint(point_preload, hashsteps, c));

		thrust::host_vector<wide_as_pair_GPU> op_hv = output_GPU;
		std::vector<wide_as_pair> output(indexbank.size());
		for (int i = 0; i < indexbank.size(); i++)
		{
			output[i].first.first = op_hv[i].first.first;
			output[i].first.second = op_hv[i].first.second;
			output[i].second.first = op_hv[i].second.first;
			output[i].second.second = op_hv[i].second.second;
		}

		return output;

	}

} //namespace bitecoin


/////////////////////////////////////////////////////////////////////////////////////

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
						  uint32_t* sharedbank2,
						  uint32_t* sharedbank1,
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


void Clockwork_wrapper(uint32_t* staticbank,
						  uint32_t* regbank,
						  uint32_t* sharedbank2,
						  uint32_t* sharedbank1,
						  //uint32_t N,
						  int* bestiBuff,
						  int* bestiBuffHead,
						  int blocks,
						  int threadsPerBlock)
{
	Clockwork <128> <<<blocks, threadsPerBlock>>> (staticbank, regbank, sharedbank2, sharedbank1, bestiBuff, bestiBuffHead);
}