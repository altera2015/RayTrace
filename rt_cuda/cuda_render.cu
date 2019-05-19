#include <stdio.h>
#include <sstream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "color.h"
#include "vec3.h"
#include "ray.h"
#include "rnd.h"
#include "camera.h"
#include "hitable.h"
#include "scenes.h"
#include "material.h"
#include "sphere.h"
#include "../rt/memorybuffer.h"
#include "../rt/dostream.h"

static dostream dbg;

#include "cuda_render.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}


__device__ RGBColor color(ray & r, const hitable & world, int depth, Rnd & rnd)
{
	hit_record rec;
	RGBColor attenuated(1.0f, 1.0f, 1.0f);
	ray scattered;
	for (int i = 0; i < depth; i++)
	{
		if (world.hit(r, 0.0001f, 10000.0f, rec))
		{
			ray scattered;
			RGBColor a;
			if (rec.mat_ptr->scatter(r, rec, a, scattered, rnd))			
			{				
				attenuated *= a;
				r = scattered;
			}
			else
			{
				return RGBColor();
			}
		}
		else
		{
			vec3 dir = unit_vector(r.direction());
			float t = 0.5f * (dir.y() + 1.0f);
			RGBColor background = (1.0f - t) * RGBColor(1.0f, 1.0f, 1.0f) + t * RGBColor(0.5f, 0.7f, 1.0f);			
			return background * attenuated;
		}
	}

	return RGBColor();
}


#define BAKED_LOOPS 10
__global__ void renderblock(RGBColor *fb, int max_x, int max_y, int startBlock, int max_bounces, camera cam, Rnd rnd, hitable & world, BlockCalc bcalc)
{
	int block = startBlock + blockIdx.x;
	int i=0, j=0;
	bcalc.pixelOffset(block, i, j);

	i += threadIdx.x;
	j += threadIdx.y;

	if ((i >= max_x) || (j >= max_y))
	{
		return;
	}

	int pixel_index = j * max_x + i;

	RGBColor col;
	#pragma unroll
	for (int sample = 0; sample < BAKED_LOOPS; sample++)
	{
		float u = float(i + rnd.random()) / float(max_x);
		float v = float(j + rnd.random()) / float(max_y);

		ray r = cam.get_ray(u, v, rnd);
		col += color(r, world, max_bounces, rnd);
	}		

	atomicAdd(&fb[pixel_index].r, col.r);
	atomicAdd(&fb[pixel_index].g, col.g);
	atomicAdd(&fb[pixel_index].b, col.b);
}


CudaRender::CudaRender(int blocksizeX, int blocksizeY, int blocksPerCompute, int samples) :	
	_blocks(blocksPerCompute),
	_threads(blocksizeX, blocksizeY, 32),
	_samples(samples)
{	
	int device;
	checkCudaErrors(cudaGetDevice(&device));
	
	int maxThreadsPerBlock;
	checkCudaErrors(cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device));

	int multiProcessors;
	checkCudaErrors(cudaDeviceGetAttribute(&multiProcessors, cudaDevAttrMultiProcessorCount, device));

	if (blocksizeY < 0 || blocksizeX <0 || blocksPerCompute < 0 ) {
		blocksPerCompute = multiProcessors;
		blocksizeX = maxThreadsPerBlock / 32;
		blocksizeY = 1;
		_blocks = blocksPerCompute;
		_threads = dim3(blocksizeX, blocksizeY, 32);		
	}

	dbg << "Using " << blocksizeX * 32 << " threads and " << blocksPerCompute << " computes." << std::endl;
	_blockCalculator.setBlockSize(blocksizeX, blocksizeY);
	_rnd = new Rnd(_blocks, _threads);
}

int CudaRender::setup(RGB8MemoryBuffer * buffer, camera * cam)
{
	_buffer = buffer;
	_cam = cam;
	_world = random_scene();
	_currentBlock = 0;		
	_blockCalculator.setImage(buffer->width(), buffer->height());	
	size_t pixelCount = buffer->length();
	size_t byteSize = sizeof(RGBColor) * pixelCount;
	checkCudaErrors(cudaMallocManaged(&_fbdev, byteSize));
	for (int j = 0; j < _buffer->height(); j++)
	{
		for (int i = 0; i < _buffer->width(); i++)
		{
			int pixel_index = j * _buffer->width() + i;
			_fbdev[pixel_index] = RGBColor(0.0f, 0.0f, 0.0f);
		}
	}
	cudaDeviceSynchronize();
	return _blockCalculator.blockSum();
}

int CudaRender::computeNext()
{
	int i, j;
	_blockCalculator.pixelOffset(_currentBlock, i, j);
	unsigned int totalLoops = _samples / BAKED_LOOPS;

	while (totalLoops > 0)
	{
		if (totalLoops < _threads.z)
		{
			_threads.z = totalLoops;
		}
		renderblock << <_blocks, _threads >> > (_fbdev, _buffer->width(), _buffer->height(), _currentBlock,
			20,
			*_cam,
			*_rnd,
			*_world,
			_blockCalculator);
		
		totalLoops -= _threads.z;
	}

	cudaDeviceSynchronize();
	_currentBlock += _blocks.x * _blocks.y;
	return _currentBlock;
}


bool CudaRender::syncBuffers()
{	
	size_t pixelCount = _buffer->length();
	RGB8Color * o = _buffer->raw();

	for (int j = 0; j < _buffer->height(); j++)
	{
		for (int i = 0; i < _buffer->width(); i++)
		{
			int pixel_index = j * _buffer->width() + i;
			RGBColor & col = _fbdev[pixel_index];
			o[pixel_index] = RGB8Color(uint8_t(sqrt(col.r/100.0f) * 255.99f), uint8_t(sqrt(col.g/100.0f) * 255.99f), uint8_t(sqrt(col.b/100.0f) * 255.99f));
		}
	}

	// probably should only copy the updated part of the image
	// cudaMemcpy(_buffer->raw(), _fbdev, byteSize, cudaMemcpyDeviceToHost);
	return true;
}

bool CudaRender::cleanup()
{
	cudaDeviceSynchronize();	
	delete _rnd;
	_rnd = nullptr;
	delete _world;
	_world = nullptr;
	if (_fbdev != nullptr)
	{
		cudaFree(_fbdev);
	}
	_fbdev = nullptr;
	return true;
}


std::string CudaRender::info()
{
	std::string result;
	int device;
	checkCudaErrors(cudaGetDevice(&device));

	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, device));

	std::stringstream ss;
	ss << "CUDA Renderer. Using " <<prop.name << " with Compute " << prop.major << "." << prop.minor << " SMP Count = " << prop.multiProcessorCount << ", Max Threads per SMP = " << prop.maxThreadsPerMultiProcessor;
	return ss.str();
}
