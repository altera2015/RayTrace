#include "rnd.h"
#include <cuda_runtime.h>
//#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>


__global__ void rnd_init(size_t _count, curandState * _state)
{
	int index = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + (blockIdx.x * blockDim.x * blockDim.y * blockDim.z);
	if (index >= _count)
	{
		return;
	}
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, index, 0, &_state[index]);
}

__host__ Rnd::Rnd(const Rnd & other) : _owner(false)
{
	_state = other._state;
	_count = other._count;	
}

__host__ Rnd::Rnd(dim3 blocks, dim3 threads) : _state(nullptr), _owner(true)
{		
	_count = blocks.x * blocks.y * threads.x * threads.y * threads.z;
	cudaMalloc(&_state, _count * sizeof(curandState));
	rnd_init <<<blocks, threads>>> (_count, _state);
}

__host__ Rnd::~Rnd()
{
	if (_owner)
	{
		cudaFree(&_state);
	}
}

__device__ float Rnd::random()
{
	int index = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + (blockIdx.x * blockDim.x * blockDim.y * blockDim.z);
	if (index >= _count)
	{
		return 0;
	}
	return curand_uniform(&_state[index]);
}

__device__ vec3 Rnd::random_in_unit_disk()
{
	vec3 p;
	do {
		p = 2.0*vec3(random(), random(), 0) - vec3(1, 1, 0);
	} while (dot(p, p) >= 1.0);
	return p;
}

__device__ vec3 Rnd::random_in_unit_sphere()
{
	vec3 p;
	do {
		p = 2.0*vec3(random(), random(), random()) - vec3(1, 1, 1);
	} while (p.squared_length() >= 1.0);
	return p;
}


