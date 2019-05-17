#ifndef __RND_H__
#define __RND_H__

#include <random>
#include "vec3.h"
#include <cuda.h>
#include <curand_kernel.h>

class Rnd {

	curandState * _state;
	size_t _pixels;
	int _width;

public:
	__host__ __device__ Rnd(size_t pixels, int width);

	__host__ void Rnd::init(dim3 blocks, dim3 threads);
	__host__ void dealloc();
	
	__device__ float random();
	__device__ vec3 random_in_unit_disk();
	__device__ vec3 random_in_unit_sphere();
};

#endif