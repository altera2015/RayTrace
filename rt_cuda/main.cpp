// Add /openmp to VS compiler to run in parallel!

// Baseline:
// 93 seconds to render random_scene with 100 samples and 20 bounces at 400 x 200

// OpenMP with 4 core i-Core 7
// 22 seconds to render random_scene with 100 samples and 20 bounces at 400 x 200

// Cuda
// 3.5 seconds

// RTX

#include "ray.h"
#include "cuda_render.h"

#include <chrono>
#include <ctime> 

#include "../rt/dostream.h"
#include "../rt/sdlhelpers.h"
#include "../rt/memorybuffer.h"

static dostream dbg;

int runMain()
{
	const int width = 400;
	const int height = 200;

	SDL_WindowPtr win;
	SDL_RendederPtr ren;

	RGB8MemoryBuffer mb(width, height, RGB8Color(0x00, 0xff, 0xff));
	CudaRender r;
	dbg << r.info() << std::endl;

	vec3 look_from(13.0f, 2.0f, 3.0f);	
	vec3 look_at(0.0f, 0.0f, 0.0f);
	vec3 camera_up(0.0f, 1.0f, 0.0f);

	camera cam(look_from, look_at, camera_up, 25, float(mb.width()) / float(mb.height()), 0.1f, (look_at - look_from).length());

	int totalBlocks = 0;
	totalBlocks = r.setup(&mb, &cam);

	if (!SDLH_SetupWindow(100, 100, width, height, win, ren))
	{
		return -1;
	}

	SDL_SurfacePtr surface = SDLH_SurfaceFromMemoryBuffer(mb);
	if (surface == nullptr)
	{
		dbg << "SDL_CreateRGBSurfaceFrom Error: " << SDL_GetError() << std::endl;
		return -4;
	}

	auto start = std::chrono::system_clock::now();

	int loop = 0;
	int block = 0;
	SDL_Event e;
	bool quit = false;
	while (!quit) {

		while (SDL_PollEvent(&e)) {
			if (e.type == SDL_QUIT) {
				quit = true;
			}
			if (e.type == SDL_KEYDOWN) {
				quit = true;
			}
			if (e.type == SDL_MOUSEBUTTONDOWN) {
				quit = true;
			}
		}

		std::chrono::duration<double> totalComputeTime;
		if (block < totalBlocks )
		{
			auto step_start = std::chrono::system_clock::now();			
			block = r.computeNext();			
			auto end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_seconds = end - step_start;
			totalComputeTime += elapsed_seconds;
			dbg << "Step took " << (elapsed_seconds.count()) << " seconds" << std::endl;
			r.syncBuffers();
			if (block >= totalBlocks)
			{				
				dbg << "Cleanup called." << std::endl;
				r.cleanup();
				mb_save(mb, "test.png");
				auto end = std::chrono::system_clock::now();				
				dbg << "Rendering took " << (totalComputeTime.count()) << " seconds" << std::endl;

			}
		}		

		SDL_RenderClear(ren.get());

		SDL_TexturePtr tex(SDL_CreateTextureFromSurface(ren.get(), surface.get()));
		if (tex == nullptr)
		{
			dbg << "SDL_CreateTextureFromSurface Error: " << SDL_GetError() << std::endl;
			return -5;
		}

		SDL_RenderCopy(ren.get(), tex.get(), NULL, NULL);

		SDL_RenderPresent(ren.get());

	}
	return 0;
}


int __stdcall WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd)
{
	if (SDL_Init(SDL_INIT_VIDEO) != 0)
	{
		dbg << "SDL_Init Error: " << SDL_GetError() << std::endl;
		return -1;
	}

	int res = runMain();

	SDL_Quit();

	return res;
}
