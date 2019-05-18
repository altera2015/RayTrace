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

__global__ void renderblock(RGB8Color *fb, int max_x, int max_y, int startBlock, int samples, int max_bounces, camera cam, Rnd rnd, hitable & world, BlockCalc bcalc)
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
	for (int sample = 0; sample < samples; sample++)
	{
		float u = float(i + rnd.random()) / float(max_x);
		float v = float(j + rnd.random()) / float(max_y);

		ray r = cam.get_ray(u, v, rnd);
		col += color(r, world, max_bounces, rnd);
	}
	col *= (1.0f / samples);
	fb[pixel_index] = RGB8Color(uint8_t(sqrt(col.r) * 255.99f), uint8_t(sqrt(col.g) * 255.99f), uint8_t(sqrt(col.b) * 255.99f));
}


CudaRender::CudaRender(int blocksizeX, int blockSizeY, int blocksPerCompute) :	
	_blocks(blocksPerCompute),
	_threads(blocksizeX, blockSizeY)
{	
	_blockCalculator.setBlockSize(blocksizeX, blockSizeY);
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
	size_t byteSize = sizeof(RGB8Color) * pixelCount;
	checkCudaErrors(cudaMalloc(&_fbdev, byteSize));
	cudaDeviceSynchronize();
	return _blockCalculator.blockSum();
}

int CudaRender::computeNext()
{
	int i, j;
	_blockCalculator.pixelOffset(_currentBlock, i, j);

	renderblock<<<_blocks, _threads >>>(_fbdev, _buffer->width(), _buffer->height(), _currentBlock,
		100,
		20,
		*_cam,
		*_rnd,
		*_world,
		_blockCalculator);

	cudaDeviceSynchronize();
	_currentBlock += _blocks.x * _blocks.y;
	return _currentBlock;
}


bool CudaRender::syncBuffers()
{	
	size_t pixelCount = _buffer->length();
	size_t byteSize = sizeof(RGB8Color) * pixelCount;
	
	// probably should only copy the updated part of the image
	cudaMemcpy(_buffer->raw(), _fbdev, byteSize, cudaMemcpyDeviceToHost);
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
