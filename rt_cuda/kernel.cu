#include <stdio.h>

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
			if (rec.mat_ptr == nullptr) {
				return RGBColor(1.0f, 0.0f, 0.0f);
			}
			
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


__global__ void render(RGB8Color *fb, int max_x, int max_y, int samples, int max_bounces, camera cam, Rnd  rnd, hitable & world)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
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

		ray r = cam.get_ray(u, v);
		col += color(r, world, max_bounces, rnd);
	}
	col *= (1.0f / samples);	
	fb[pixel_index] = RGB8Color(uint8_t(sqrt(col.r) * 255.99f), uint8_t(sqrt(col.g) * 255.99f), uint8_t(sqrt(col.b) * 255.99f));
}

bool render_cuda(RGB8MemoryBuffer & mb)
{
	int tx = 8;
	int ty = 8;

	dim3 blocks(mb.width() / tx + 1, mb.height() / ty + 1);
	dim3 threads(tx, ty);

	Rnd rnd(blocks, threads, mb.length(), mb.width());

	size_t pixelCount = mb.length();
	size_t byteSize = sizeof(RGB8Color) * pixelCount;
	
	vec3 look_from(13.0f, 2.0f, 3.0f);
	// vec3 look_at(0.0f, 0.0f, -1.0f);
	vec3 look_at(0.0f, 0.0f, 0.0f);
	vec3 camera_up(0.0f, 1.0f, 0.0f);

	camera cam(look_from, look_at, camera_up, 25, float(mb.width()) / float(mb.height()), 0.1f, (look_at - look_from).length(), rnd);

	RGB8Color * fb_dev = nullptr;
	checkCudaErrors(cudaMalloc(&fb_dev, byteSize));
	
	hitable * world = buildWorld();
	cudaDeviceSynchronize();

	render<<<blocks, threads>>>(fb_dev, mb.width(), mb.height(),
		100,
		20,
		cam,
		rnd,
		*world);
		
	cudaDeviceSynchronize();
	
	cudaMemcpy(mb.raw(), fb_dev, byteSize, cudaMemcpyDeviceToHost);
	cudaFree(fb_dev);

	// delete world;
	
	return true;
}