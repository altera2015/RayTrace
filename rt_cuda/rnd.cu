#include "rnd.h"
#include <cuda_runtime.h>
//#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>


__global__ void rnd_init(size_t _pixels, int _width, curandState * _state)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	size_t pixel_index = j * _width + i;
	if (pixel_index >= _pixels)
	{
		return;
	}
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &_state[pixel_index]);
}

__host__ Rnd::Rnd(const Rnd & other) : _owner(false)
{
	_state = other._state;
	_pixels = other._pixels;
	_width = other._width;
}

__host__ Rnd::Rnd(dim3 blocks, dim3 threads, size_t pixels, int width) : _pixels(pixels), _width(width), _state(nullptr), _owner(true)
{	
	cudaMalloc(&_state, _pixels * sizeof(curandState));
	rnd_init <<<blocks, threads>>> (_pixels, _width, _state);
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
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int pixel_index = j * _width + i;
	if (pixel_index >= _pixels)
	{
		return 0.0f;
	}
	return curand_uniform(&_state[pixel_index]);
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


