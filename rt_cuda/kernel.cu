#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "color.h"
#include "vec3.h"
#include "ray.h"
#include "rnd.h"
#include "camera.h"

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

__device__ RGBColor color(const ray& r) {
	vec3 unit_direction = unit_vector(r.direction());
	float t = 0.5f*(unit_direction.y() + 1.0f);
	return (1.0f - t)*RGBColor(1.0f, 1.0f, 1.0f) + t * RGBColor(0.5f, 0.7f, 1.0f);
}

__global__ void render(RGB8Color *fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin, Rnd rnd, camera cam)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y))
	{
		return;
	}

	int pixel_index = j * max_x + i;
	float u = float(i) / float(max_x);
	float v = float(j) / float(max_y);
	// ray r(origin, lower_left_corner + u * horizontal + v * vertical);
	ray r = cam.get_ray_pinhole(u, v);
	RGBColor col = color(r);
	fb[pixel_index] = RGB8Color(uint8_t(sqrt(col.r) * 255.99f), uint8_t(sqrt(col.g) * 255.99f), uint8_t(sqrt(col.b) * 255.99f));
}

bool render_cuda(RGB8MemoryBuffer & mb)
{
	Rnd rnd(mb.length(), mb.width());
	camera cam(
		vec3(0.0, 0.0, -2.0),
		vec3(0.0, 0.0, 0.0),
		vec3(0.0, 1.0, 0.0),
		20.0f,
		float(mb.width()) / float(mb.height()),
		0.0f,


	);

	int tx = 8;
	int ty = 8;

	dim3 blocks(mb.width() / tx + 1, mb.height() / ty + 1);
	dim3 threads(tx, ty);

	rnd.init(blocks, threads);

	size_t pixelCount = mb.length();
	size_t byteSize = sizeof(RGB8Color) * pixelCount;

	RGB8Color * fb_dev = nullptr;
	checkCudaErrors(cudaMalloc(&fb_dev, byteSize));
	
	render<<<blocks, threads>>> (fb_dev, mb.width(), mb.height(),
		vec3(-2.0, -1.0, -1.0),
		vec3(4.0, 0.0, 0.0),
		vec3(0.0, 2.0, 0.0),
		vec3(0.0, 0.0, 0.0),
		rnd);

	cudaDeviceSynchronize();
	
	cudaMemcpy(mb.raw(), fb_dev, byteSize, cudaMemcpyDeviceToHost);
	cudaFree(fb_dev);

	rnd.dealloc();
	return true;
}