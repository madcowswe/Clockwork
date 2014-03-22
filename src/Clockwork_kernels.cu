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
#include <thrust/sort.h>



typedef std::pair<std::pair<uint64_t, uint64_t>, std::pair<uint64_t, uint64_t> >
	wide_as_pair;
typedef std::pair<wide_as_pair, std::vector<uint32_t> >
	point_idx_pair;
typedef thrust::pair<thrust::pair<uint64_t, uint64_t>, thrust::pair<uint64_t, uint64_t> >
	wide_as_pair_GPU;

struct bigint_t
{
	uint32_t limbs[8];
};

struct halfbigint_t
{
	uint32_t limbs[4];
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
	const unsigned diff;
	uint32_t c[4];

	genpoint(const uint32_t* const ppre_in, unsigned ns_in, const uint32_t* const c_in, unsigned diff_in) :
		numsteps(ns_in), diff(diff_in)
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

		uint32_t point[2][8];

		for (int isdiff = 0; isdiff <= 1 ; isdiff++)
		{
			point[isdiff][0] = idx + isdiff*diff;
			for (int i = 1; i < 8; i++)
			{
				point[isdiff][i] = point_preload[i];
			}

			for (int i = 0; i < numsteps; i++)
			{
				bigint_t tmp;
				// tmp=lo(x)*c;
				wide_mul_GPU(4, tmp.limbs+4, tmp.limbs, point[isdiff], c);
				// [carry,lo(x)] = lo(tmp)+hi(x)
				uint32_t carry=wide_add_GPU(4, point[isdiff], tmp.limbs, point[isdiff]+4);
				// hi(x) = hi(tmp) + carry
				wide_add_GPU(4, point[isdiff]+4, tmp.limbs+4, carry);

				// overall:  tmp=lo(x)*c; x=tmp+hi(x)
			}

		}

		uint32_t mpoint[8];
		for (int i = 0; i < 8; i++)
		{
			mpoint[i] = point[0][i] ^ point[1][i];
		}

		return thrust::make_pair(
			thrust::make_pair(
			((uint64_t)mpoint[7] << 32) + mpoint[6], 
			((uint64_t)mpoint[5] << 32) + mpoint[4]
		),
			thrust::make_pair(
			((uint64_t)mpoint[3] << 32) + mpoint[2], 
			((uint64_t)mpoint[1] << 32) + mpoint[0]
		));

	}
};



//modified from https://devtalk.nvidia.com/default/topic/610914/cuda-programming-and-performance/modular-exponentiation-amp-biginteger/
__device__ __forceinline__ bigint_t add256 (bigint_t a, bigint_t b)
{
    bigint_t res;
    asm ("{\n\t"
         "add.cc.u32      %0,  %8, %16; \n\t"
         "addc.cc.u32     %1,  %9, %17; \n\t"
         "addc.cc.u32     %2, %10, %18; \n\t"
         "addc.cc.u32     %3, %11, %19; \n\t"
         "addc.cc.u32     %4, %12, %20; \n\t"
         "addc.cc.u32     %5, %13, %21; \n\t"
         "addc.cc.u32     %6, %14, %22; \n\t"
         "addc.u32        %7, %15, %23; \n\t"
         "}"
         : "=r"(res.limbs[0]), "=r"(res.limbs[1]), "=r"(res.limbs[2]), "=r"(res.limbs[3]), "=r"(res.limbs[4]), "=r"(res.limbs[5]), "=r"(res.limbs[6]), "=r"(res.limbs[7])
         : "r"(a.limbs[0]), "r"(a.limbs[1]), "r"(a.limbs[2]), "r"(a.limbs[3]),"r"(a.limbs[4]), "r"(a.limbs[5]), "r"(a.limbs[6]), "r"(a.limbs[7]),
           "r"(b.limbs[0]), "r"(b.limbs[1]), "r"(b.limbs[2]), "r"(b.limbs[3]),"r"(b.limbs[4]), "r"(b.limbs[5]), "r"(b.limbs[6]), "r"(b.limbs[7]));
    return res;
}

//modified from https://devtalk.nvidia.com/default/topic/610914/cuda-programming-and-performance/modular-exponentiation-amp-biginteger/
__device__ __forceinline__ bigint_t add256 (bigint_t a, halfbigint_t b)
{
    bigint_t res;
    asm ("{\n\t"
         "add.cc.u32      %0,  %8, %16; \n\t"
         "addc.cc.u32     %1,  %9, %17; \n\t"
         "addc.cc.u32     %2, %10, %18; \n\t"
         "addc.cc.u32     %3, %11, %19; \n\t"
         "addc.u32        %4, %12,   0; \n\t"
         //"addc.cc.u32     %5, %13, %21; \n\t"
		 "mov.u32         %5, %13     ; \n\t"
         //"addc.cc.u32     %6, %14, %22; \n\t"
		 "mov.u32         %6, %14     ; \n\t"
         //"addc.u32        %7, %15, %23; \n\t"
		 "mov.u32         %7, %15     ; \n\t"
         "}"
         : "=r"(res.limbs[0]), "=r"(res.limbs[1]), "=r"(res.limbs[2]), "=r"(res.limbs[3]), "=r"(res.limbs[4]), "=r"(res.limbs[5]), "=r"(res.limbs[6]), "=r"(res.limbs[7])
         : "r"(a.limbs[0]), "r"(a.limbs[1]), "r"(a.limbs[2]), "r"(a.limbs[3]),"r"(a.limbs[4]), "r"(a.limbs[5]), "r"(a.limbs[6]), "r"(a.limbs[7]),
           "r"(b.limbs[0]), "r"(b.limbs[1]), "r"(b.limbs[2]), "r"(b.limbs[3]));
    return res;
}

//modified from https://devtalk.nvidia.com/default/topic/610914/cuda-programming-and-performance/modular-exponentiation-amp-biginteger/
__device__ __forceinline__ bigint_t umul256 (bigint_t a, bigint_t b)
{
    bigint_t res;
    //top 12 13 14 15, 20 21 22 23
    asm ("{\n\t"
         "mul.lo.u32      %0,  %8, %16;    \n\t"

         "mul.hi.u32      %1,  %8, %16;    \n\t"

         "mad.lo.cc.u32   %1,  %8, %17, %1;\n\t"
         "madc.hi.u32     %2,  %8, %17,  0;\n\t"

         "mad.lo.cc.u32   %1,  %9, %16, %1;\n\t"
         "madc.hi.cc.u32  %2,  %9, %16, %2;\n\t"
         "madc.hi.u32     %3,  %8, %18,  0;\n\t"

         "mad.lo.cc.u32   %2,  %8, %18, %2;\n\t"
         "madc.hi.cc.u32  %3,  %9, %17, %3;\n\t"
         "madc.hi.u32     %4,  %8, %19,  0;\n\t"

         "mad.lo.cc.u32   %2,  %9, %17, %2;\n\t"
         "madc.hi.cc.u32  %3, %10, %16, %3;\n\t"
         "madc.hi.cc.u32  %4,  %9, %18, %4;\n\t"
         "madc.hi.u32     %5,  %8, %20,  0;\n\t" // Uses top

         "mad.lo.cc.u32   %2, %10, %16, %2;\n\t"
         "madc.lo.cc.u32  %3,  %8, %19, %3;\n\t"
         "madc.hi.cc.u32  %4, %10, %17, %4;\n\t"
         "madc.hi.cc.u32  %5,  %9, %19, %5;\n\t"
         "madc.hi.u32     %6,  %8, %21,  0;\n\t" // Uses top

         "mad.lo.cc.u32   %3,  %9, %18, %3;\n\t"
         "madc.hi.cc.u32  %4, %11, %16, %4;\n\t"
         "madc.hi.cc.u32  %5, %10, %18, %5;\n\t"
         "madc.hi.cc.u32  %6,  %9, %20, %6;\n\t" // Uses top
         "madc.hi.u32     %7,  %8, %22,  0;\n\t" // Uses top

         "mad.lo.cc.u32   %3, %10, %17, %3;\n\t"
         "madc.lo.cc.u32  %4,  %8, %20, %4;\n\t" // Uses top
         "madc.hi.cc.u32  %5, %11, %17, %5;\n\t"
         "madc.hi.cc.u32  %6, %10, %19, %6;\n\t"
         "madc.hi.u32     %7,  %9, %21, %7;\n\t" // Uses top

         "mad.lo.cc.u32   %3, %11, %16, %3;\n\t"
         "madc.lo.cc.u32  %4,  %9, %19, %4;\n\t"
         "madc.hi.cc.u32  %5, %12, %16, %5;\n\t" // Uses top
         "madc.hi.cc.u32  %6, %11, %18, %6;\n\t"
         "madc.hi.u32     %7, %10, %20, %7;\n\t" // Uses top

         "mad.lo.cc.u32   %4, %10, %18, %4;\n\t"
         "madc.lo.cc.u32  %5,  %8, %21, %5;\n\t" // Uses top
         "madc.hi.cc.u32  %6, %12, %17, %6;\n\t" // Uses top
         "madc.hi.u32     %7, %11, %19, %7;\n\t"

         "mad.lo.cc.u32   %4, %11, %17, %4;\n\t"
         "madc.lo.cc.u32  %5,  %9, %20, %5;\n\t" // Uses top
         "madc.hi.cc.u32  %6, %13, %16, %6;\n\t" // Uses top
         "madc.hi.u32     %7, %12, %18, %7;\n\t" // Uses top

         "mad.lo.cc.u32   %4, %12, %16, %4;\n\t" // Uses top
         "madc.lo.cc.u32  %5, %10, %19, %5;\n\t"
         "madc.lo.cc.u32  %6,  %8, %22, %6;\n\t" // Uses top
         "madc.hi.u32     %7, %13, %17, %7;\n\t" // Uses top

         "mad.lo.cc.u32   %5, %11, %18, %5;\n\t"
         "madc.lo.cc.u32  %6,  %9, %21, %6;\n\t" // Uses top
         "madc.hi.u32     %7, %14, %16, %7;\n\t" // Uses top

         "mad.lo.cc.u32   %5, %12, %17, %5;\n\t" // Uses top
         "madc.lo.cc.u32  %6, %10, %20, %6;\n\t" // Uses top
         "madc.lo.u32     %7,  %8, %23, %7;\n\t" // Uses top

         "mad.lo.cc.u32   %5, %13, %16, %5;\n\t" // Uses top
         "madc.lo.cc.u32  %6, %11, %19, %6;\n\t"
         "madc.lo.u32     %7,  %9, %22, %7;\n\t" // Uses top

         "mad.lo.cc.u32   %6, %12, %18, %6;\n\t" // Uses top
         "madc.lo.u32     %7, %10, %21, %7;\n\t" // Uses top

         "mad.lo.cc.u32   %6, %13, %17, %6;\n\t" // Uses top
         "madc.lo.u32     %7, %11, %20, %7;\n\t" // Uses top

         "mad.lo.cc.u32   %6, %14, %16, %6;\n\t" // Uses top
         "madc.lo.u32     %7, %12, %19, %7;\n\t" // Uses top

         "mad.lo.u32      %7, %13, %18, %7;\n\t" // Uses top

         "mad.lo.u32      %7, %14, %17, %7;\n\t" // Uses top
         
         "mad.lo.u32      %7, %15, %16, %7;\n\t" // Uses top
         "}"
         : "=r"(res.limbs[0]), "=r"(res.limbs[1]), "=r"(res.limbs[2]), "=r"(res.limbs[3]), "=r"(res.limbs[4]), "=r"(res.limbs[5]), "=r"(res.limbs[6]), "=r"(res.limbs[7])
         : "r"(a.limbs[0]), "r"(a.limbs[1]), "r"(a.limbs[2]), "r"(a.limbs[3]),"r"(a.limbs[4]), "r"(a.limbs[5]), "r"(a.limbs[6]), "r"(a.limbs[7]),
           "r"(b.limbs[0]), "r"(b.limbs[1]), "r"(b.limbs[2]), "r"(b.limbs[3]),"r"(b.limbs[4]), "r"(b.limbs[5]), "r"(b.limbs[6]), "r"(b.limbs[7]));
    return res;
}

//modified from https://devtalk.nvidia.com/default/topic/610914/cuda-programming-and-performance/modular-exponentiation-amp-biginteger/
__device__ __forceinline__ bigint_t umul_256_128in (halfbigint_t a, halfbigint_t b)
{
    bigint_t res;
    //old top 12 13 14 15, 20 21 22 23
    asm ("{\n\t"
         "mul.lo.u32      %0,  %8, %12;    \n\t"

         "mul.hi.u32      %1,  %8, %12;    \n\t"

         "mad.lo.cc.u32   %1,  %8, %13, %1;\n\t"
         "madc.hi.u32     %2,  %8, %13,  0;\n\t"

         "mad.lo.cc.u32   %1,  %9, %12, %1;\n\t"
         "madc.hi.cc.u32  %2,  %9, %12, %2;\n\t"
         "madc.hi.u32     %3,  %8, %14,  0;\n\t"

         "mad.lo.cc.u32   %2,  %8, %14, %2;\n\t"
         "madc.hi.cc.u32  %3,  %9, %13, %3;\n\t"
         "madc.hi.u32     %4,  %8, %15,  0;\n\t"

         "mad.lo.cc.u32   %2,  %9, %13, %2;\n\t"
         "madc.hi.cc.u32  %3, %10, %12, %3;\n\t"
         "madc.hi.cc.u32  %4,  %9, %14, %4;\n\t"
         //"madc.hi.u32     %5,  %8,   0,  0;\n\t" // Uses top
           "addc.u32      %5,0,0;\n\t"

         "mad.lo.cc.u32   %2, %10, %12, %2;\n\t"
         "madc.lo.cc.u32  %3,  %8, %15, %3;\n\t"
         "madc.hi.cc.u32  %4, %10, %13, %4;\n\t"
         "madc.hi.cc.u32  %5,  %9, %15, %5;\n\t"
         //"madc.hi.u32     %6,  %8,   0,  0;\n\t" // Uses top
           "addc.u32      %6,0,0;\n\t"

         "mad.lo.cc.u32   %3,  %9, %14, %3;\n\t"
         "madc.hi.cc.u32  %4, %11, %12, %4;\n\t"
         "madc.hi.cc.u32  %5, %10, %14, %5;\n\t"
         //"madc.hi.cc.u32  %6,  %9,   0,  0;\n\t" // Uses top
           "addc.u32      %6,%6,0;\n\t"
         //"madc.hi.u32     %7,  %8,   0,  0;\n\t" // Uses top

         "mad.lo.cc.u32   %3, %10, %13, %3;\n\t"
         //"madc.lo.cc.u32  %4,  %8,   0, %4;\n\t" // Uses top
           "addc.u32      %4,%4,0;\n\t"

         "mad.hi.cc.u32   %5, %11, %13, %5;\n\t"
         "madc.hi.cc.u32  %6, %10, %15, %6;\n\t"
         //"madc.hi.u32     %7,  %9,   0, %7;\n\t" // Uses top
           "addc.u32      %7,0,0;\n\t"

         "mad.lo.cc.u32   %3, %11, %12, %3;\n\t"
         "madc.lo.cc.u32  %4,  %9, %15, %4;\n\t"
         //"madc.hi.cc.u32  %5,   0, %12, %5;\n\t" // Uses top
           "addc.u32      %5,%5,0;\n\t"
         "mad.hi.cc.u32  %6, %11, %14, %6;\n\t"
         //"madc.hi.u32     %7, %10,   0, %7;\n\t" // Uses top
     	   "addc.u32      %7,%7,0;\n\t"

         "mad.lo.cc.u32   %4, %10, %14, %4;\n\t"
         //"madc.lo.cc.u32  %5,  %8,   0, %5;\n\t" // Uses top
           "addc.u32      %5,%5,0;\n\t"
         //"madc.hi.cc.u32  %6,   0, %13, %6;\n\t" // Uses top
         "mad.hi.u32     %7, %11, %15, %7;\n\t"

         "mad.lo.cc.u32   %4, %11, %13, %4;\n\t"
         //"madc.lo.cc.u32  %5,  %9,   0, %5;\n\t" // Uses top
           "addc.u32      %5,%5,0;\n\t"
         //"madc.hi.cc.u32  %6,   0, %12, %6;\n\t" // Uses top
         //"madc.hi.u32     %7,   0, %14, %7;\n\t" // Uses top

         //"mad.lo.cc.u32   %4,   0, %12, %4;\n\t" // Uses top
         "mad.lo.cc.u32   %5, %10, %15, %5;\n\t"
         //"madc.lo.cc.u32  %6,  %8,   0, %6;\n\t" // Uses top
           "addc.u32      %6,%6,0;\n\t"
         //"madc.hi.u32     %7,   0, %13, %7;\n\t" // Uses top

         "mad.lo.cc.u32   %5, %11, %14, %5;\n\t"
         //"madc.lo.cc.u32  %6,  %9,   0, %6;\n\t" // Uses top
           "addc.u32      %6,%6,0;\n\t"
         //"madc.hi.u32     %7,   0, %12, %7;\n\t" // Uses top

         //"mad.lo.cc.u32   %5,   0, %13, %5;\n\t" // Uses top
         //"madc.lo.cc.u32  %6, %10,   0, %6;\n\t" // Uses top
         //"madc.lo.u32     %7,  %8,   0, %7;\n\t" // Uses top

         //"mad.lo.cc.u32   %5,   0, %12, %5;\n\t" // Uses top
         "mad.lo.cc.u32  %6, %11, %15, %6;\n\t"
         //"madc.lo.u32     %7,  %9,   0, %7;\n\t" // Uses top
           "addc.u32      %7,%7,0;\n\t"

         //"mad.lo.cc.u32   %6,   0, %14, %6;\n\t" // Uses top
         //"madc.lo.u32     %7, %10,   0, %7;\n\t" // Uses top

         //"mad.lo.cc.u32   %6,   0, %13, %6;\n\t" // Uses top
         //"madc.lo.u32     %7, %11,   0, %7;\n\t" // Uses top

         //"mad.lo.cc.u32   %6,   0, %12, %6;\n\t" // Uses top
         //"madc.lo.u32     %7,   0, %15, %7;\n\t" // Uses top

         //"mad.lo.u32      %7,   0, %14, %7;\n\t" // Uses top

         //"mad.lo.u32      %7,   0, %13, %7;\n\t" // Uses top

         //"mad.lo.u32      %7,   0, %12, %7;\n\t" // Uses top
         "}"
         : "=r"(res.limbs[0]), "=r"(res.limbs[1]), "=r"(res.limbs[2]), "=r"(res.limbs[3]), "=r"(res.limbs[4]), "=r"(res.limbs[5]), "=r"(res.limbs[6]), "=r"(res.limbs[7])
         : "r"(a.limbs[0]), "r"(a.limbs[1]), "r"(a.limbs[2]), "r"(a.limbs[3]),
           "r"(b.limbs[0]), "r"(b.limbs[1]), "r"(b.limbs[2]), "r"(b.limbs[3]));
    return res;
}


__global__ void genmpoints_on_GPU_fast (
	unsigned hashsteps,
	halfbigint_t c,
	bigint_t point_preload,
	uint32_t diff,
	unsigned N,
	const uint32_t* const __restrict__ indexbank,
	bigint_t* const __restrict__ mpointsout
){

	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	if(ID < N){

		bigint_t points[2];

		for (int isdiff = 0; isdiff <= 1 ; isdiff++)
		{

			points[isdiff] = point_preload;
			points[isdiff].limbs[0] = indexbank[ID] + isdiff*diff;

			for (int i = 0; i < hashsteps; i++)
			{
				
				bigint_t tmp;
				halfbigint_t lox = {{ points[isdiff].limbs[0], points[isdiff].limbs[1], points[isdiff].limbs[2], points[isdiff].limbs[3] }};
				halfbigint_t hix = {{ points[isdiff].limbs[4], points[isdiff].limbs[5], points[isdiff].limbs[6], points[isdiff].limbs[7] }};

				//bigint_t loxtest = {{points[isdiff].limbs[0], points[isdiff].limbs[1], points[isdiff].limbs[2], points[isdiff].limbs[3],  0,0,0,0 }};
				//bigint_t ctest = {{ c.limbs[0], c.limbs[1], c.limbs[2], c.limbs[3], 0,0,0,0 }};


				// tmp=lo(x)*c;
				tmp = umul_256_128in(lox, c);
				//bigint_t tmp2 = umul256(loxtest, ctest);

				//bool eq[8];
				//for (int j = 0; j < 8; j++)
				//{
				//	eq[j] = tmp.limbs[j] == tmp2.limbs[j];
				//}

				// x=tmp+hi(x)
				points[isdiff] = add256(tmp, hix);

				// overall:  tmp=lo(x)*c; x=tmp+hi(x)
			}

		}

		bigint_t mpoint;
		for (int i = 0; i < 8; i++)
		{
			mpoint.limbs[i] = points[0].limbs[i] ^ points[1].limbs[i];
		}

		mpointsout[ID] = mpoint;
	}

}

struct bigint_t_less : public thrust::binary_function<bigint_t,bigint_t,bool>
{

	const unsigned len;
	bigint_t_less(const unsigned lenin):len(lenin) {}

  /*! Function call operator. The return value is <tt>lhs < rhs</tt>.
   */
  __host__ __device__ bool operator()(const bigint_t &lhs, const bigint_t &rhs) const {
	  //return lhs < rhs;

	  for (int i = len-1; i >= 0; i--)
	  {
		  if(lhs.limbs[i] < rhs.limbs[i])
			  return true;

		  if(lhs.limbs[i] > rhs.limbs[i])
			  return false;
	  }

	  //all equal, so not strictly less than
	  return false;

  }
}; // end less

struct bigint_t_less_idx : public thrust::binary_function<uint32_t,uint32_t,bool>
{

	const unsigned len;
	const bigint_t* const theBigint;
	bigint_t_less_idx(const unsigned lenin, const bigint_t* const theBigintin):len(lenin), theBigint(theBigintin) {}

  /*! Function call operator. The return value is <tt>lhs < rhs</tt>.
   */
  __host__ __device__ bool operator()(const uint32_t &lhs, const uint32_t &rhs) const {
	  //return lhs < rhs;

	  for (int i = len-1; i >= 0; i--)
	  {
		  if(theBigint[lhs].limbs[i] < theBigint[rhs].limbs[i])
			  return true;

		  if(theBigint[lhs].limbs[i] > theBigint[rhs].limbs[i])
			  return false;
	  }

	  //all equal, so not strictly less than
	  return false;

  }
}; // end less



int testcuda();


__global__ void unzip_struct(
	const uint32_t N,
	const uint32_t depth,
	const uint32_t* const __restrict__ map,
	const bigint_t* const __restrict__ mpointsin,
	uint32_t* const __restrict__ unzipout
){
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	if(ID < N){
		unzipout[ID] = mpointsin[map[ID]].limbs[depth];
	}
}


//Diff implicit!
struct indicies
{
	uint32_t parts[8];
};

struct point_idx{
	bigint_t point;
	indicies idx;
};


//set N = N-1 for each pass!!, npop should double
__global__ void xor_points_unconditional(
	const uint32_t N,
	const uint32_t depth,
	const uint32_t nPopulated,
	const bigint_t* const __restrict__ xorin,
	bigint_t* const __restrict__ xoredout,
	const indicies* const __restrict__ idxin,
	indicies* const __restrict__ idxout
){
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	if(ID < N-1){
		for (int i = 0; i < depth; i++)
		{
			xoredout[ID].limbs[i] = xorin[ID].limbs[i] ^ xorin[ID+1].limbs[i];
		}

		int j = 0;
		for (int i = 0; i < nPopulated; i++)
		{
			idxout[ID].parts[j++] = idxin[ID].parts[i];
			idxout[ID].parts[j++] = idxin[ID+1].parts[i];
		}
	}
}


__global__ void shove_flat_idx_into_struct(
	const uint32_t N,
	const uint32_t* const __restrict__ idxin,
	indicies* const __restrict__ idxout
){
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	if(ID < N){
		idxout[ID].parts[0] = idxin[ID];
	}
};


namespace bitecoin{


	std::pair<std::vector<bigint_t>, std::vector<uint32_t>> gensort_GPU (
		const unsigned hashsteps,
		const halfbigint_t c,
		const bigint_t point_preload,
		const uint32_t diff,
		const std::vector<uint32_t> &indexbank
	){

		cudaError e;
		unsigned N = indexbank.size();

		uint32_t* idxbankGPU, *idxbankGPUout;
		if(e = cudaMalloc(&idxbankGPU, N * sizeof(uint32_t))) fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);
		if(e = cudaMalloc(&idxbankGPUout, N * sizeof(uint32_t))) fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);
		if(e = cudaMemcpy(idxbankGPU, indexbank.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice))  fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);

		// indicies* idxa, *idxb;
		// if(e = cudaMalloc(&idxa, N * sizeof(indicies))) fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);
		// if(e = cudaMalloc(&idxb, N * sizeof(indicies))) fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);
		// auto idxaptr = thrust::device_pointer_cast(idxa);
		// auto idxbptr = thrust::device_pointer_cast(idxb);

		bigint_t* mpointsGPUa, *mpointsGPUb;
		if(e = cudaMalloc(&mpointsGPUa, N * sizeof(bigint_t))) fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);
		if(e = cudaMalloc(&mpointsGPUb, N * sizeof(bigint_t))) fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);
		auto mpointsGPUtptra = thrust::device_pointer_cast(mpointsGPUa);
		auto mpointsGPUtptrb = thrust::device_pointer_cast(mpointsGPUb);

		//gen

		unsigned nblocks = std::ceil((double)N/128);
		genmpoints_on_GPU_fast <<<nblocks, 128>>> (hashsteps, c, point_preload, diff, N, idxbankGPU, mpointsGPUa);
		if(e = cudaGetLastError()) printf("Cuda error %d on line %d\n", e, __LINE__);
		//shove_flat_idx_into_struct <<<nblocks, 128>>> (N, idxbankGPU, idxa);

		//sort

		uint32_t* map;
		if(e = cudaMalloc(&map, N * sizeof(uint32_t))) fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);
		auto maptptr = thrust::device_pointer_cast(map);

		thrust::sequence(maptptr, maptptr+N);

		uint32_t* currlimb;
		if(e = cudaMalloc(&currlimb, N * sizeof(uint32_t))) fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);
		auto currlimbptr = thrust::device_pointer_cast(currlimb);

		for (int i = 0; i < 7; i++)
		{
			unzip_struct <<<nblocks, 128>>> (N,i,map,mpointsGPUa,currlimb);
			if(e = cudaGetLastError()) printf("Cuda error %d on line %d\n", e, __LINE__);
			thrust::stable_sort_by_key(currlimbptr, currlimbptr+N, maptptr);
		}

		//gather sort results
		thrust::gather(maptptr, maptptr+N, mpointsGPUtptra, mpointsGPUtptrb);

		auto idxbankGPUptr = thrust::device_pointer_cast(idxbankGPU);
		auto idxbankGPUoutptr = thrust::device_pointer_cast(idxbankGPUout);
		thrust::gather(maptptr, maptptr+N, idxbankGPUptr, idxbankGPUoutptr);

		//xor_points_unconditional(N, 8, 1, mptsGPUoutvecRaw, m2pointsGPU, idxa, idxb);
		//std::swap(idxa,idxb);
		//std::swap(m2pointsGPU, mpointsGPU);

		//DEBUG
		//std::vector<uint32_t> testmap(N);
		//if(e = cudaMemcpy(testmap.data(), map, N * sizeof(uint32_t), cudaMemcpyDeviceToHost))  fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);

		std::vector<bigint_t> mpointsHost(N);
		if(e = cudaMemcpy(mpointsHost.data(), mpointsGPUb, N * sizeof(bigint_t), cudaMemcpyDeviceToHost)) fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);
		std::vector<uint32_t> idxHost(N);
		if(e = cudaMemcpy(idxHost.data(), idxbankGPUout, N * sizeof(bigint_t), cudaMemcpyDeviceToHost)) fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);

		cudaFree(idxbankGPU);
		cudaFree(idxbankGPUout);
		//cudaFree(idxa);
		//cudaFree(idxb);
		cudaFree(mpointsGPUa);
		cudaFree(mpointsGPUb);
		cudaFree(map);
		cudaFree(currlimb);

		return std::make_pair(mpointsHost, idxHost);
	}


	std::vector<wide_as_pair> genmpoints_on_GPU (
		unsigned hashsteps,
		const uint32_t* const c,
		const uint32_t* const point_preload,
		const std::vector<uint32_t> &indexbank,
		unsigned diff
	){

		thrust::device_vector<uint32_t> indexbank_GPU = indexbank;
		thrust::device_vector<wide_as_pair_GPU> output_GPU(indexbank.size());

		thrust::transform(indexbank_GPU.begin(), indexbank_GPU.end(), output_GPU.begin(), genpoint(point_preload, hashsteps, c, diff));

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

#if 0
	wide_as_pair gensortscan_on_GPU (
		unsigned hashsteps,
		const uint32_t* const c,
		const uint32_t* const point_preload,
		const std::vector<uint32_t> &indexbank
	){

		thrust::device_vector<uint32_t> indexbank_GPU = indexbank;
		thrust::device_vector<wide_as_pair_GPU> pointbank_gpu(indexbank.size());

		thrust::transform(indexbank_GPU.begin(), indexbank_GPU.end(), output_GPU.begin(), genpoint(point_preload, hashsteps, c));

		
		

		wide_as_pair output;

		output.first.first = op_hv.first.first;
		output.first.second = op_hv.first.second;
		output.second.first = op_hv.second.first;
		output.second.second = op_hv.second.second;

		return output;

	}
#endif


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