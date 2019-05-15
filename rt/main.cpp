
// Add /openmp to VS compiler to run in parallel!

// Baseline:
// 93 seconds to render random_scene with 100 samples and 20 bounces at 400 x 200

// OpenMP with 4 core i-Core 7
// 22 seconds to render random_scene with 100 samples and 20 bounces at 400 x 200

// Cuda

// RTX

#ifdef _OPENMP 
#include <omp.h>
#endif

#include <chrono>
#include <ctime> 

#include "dostream.h"
#include "sdlhelpers.h"
#include "memorybuffer.h"

#include "ray.h"
#include "scenes.h"
#include "camera.h"
#include "rnd.h"

// #include "sphere.h"
#include "material.h"
#include "hitable_list.h"


static dostream dbg;


RGBColor color(ray & r, const hitable & world, int depth )
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
			if (rec.mat_ptr->scatter(r, rec, a, scattered))
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

int draw_lines(RGB8MemoryBuffer & mb, camera & cam, const hitable & world, Rnd & rnd, const int j_start, const int samples = 100, const int max_bounces = 50)
{	
	int lines_done = 0;
#ifdef _OPENMP
	#pragma omp parallel default(none) shared(mb, cam, rnd, lines_done)
#endif
	{
#ifdef _OPENMP
		int threads = omp_get_num_threads();		 
#else
		int threads = 1;
#endif
		int max_j = j_start + threads < mb.height() ? j_start + threads : mb.height();

#ifdef _OPENMP
		#pragma omp for schedule(dynamic) reduction(+:lines_done)
#endif
		for (int j = j_start; j < max_j; j++)
		{
			lines_done++;
			for (int i = 0; i < mb.width(); i++)
			{
				RGBColor col;
				for (int sample = 0; sample < samples; sample++)
				{
					float u = float(i + rnd.random()) / float(mb.width());
					float v = float(j + rnd.random()) / float(mb.height());

					ray r = cam.get_ray(u, v);
					col += color(r, world, max_bounces);
				}
				col *= (1.0f / samples);
				mb.put(i, j, RGB8Color(uint8_t(sqrt(col.r) * 255.99), uint8_t(sqrt(col.g) * 255.99), uint8_t(sqrt(col.b) * 255.99)));
			}
		}
		
	}

	return j_start + lines_done;	
}



int runMain()
{
	const int width = 400;
	const int height = 200;

	SDL_WindowPtr win;
	SDL_RendederPtr ren;	
		
	if (!SDLH_SetupWindow(100, 100, width, height, win, ren))
	{
		return -1;
	}

	RGB8MemoryBuffer mb(width, height, RGB8Color(0xff,0,0 ) );

	Rnd rnd;
	//HitablePtr world(buildWorld(rnd));
	HitablePtr world(random_scene(rnd));
	

	vec3 look_from(13.0f, 2.0f, 3.0f);
	vec3 look_at(0.0f, 0.0f, -1.0f);
	vec3 camera_up(0.0f, 1.0f, 0.0f);

	camera cam(look_from, look_at, camera_up, 25, float(width) / float(height), 0.1f, (look_at - look_from).length(), rnd);
	

	SDL_SurfacePtr surface = SDLH_SurfaceFromMemoryBuffer(mb);	
	if (surface == nullptr) 
	{
		dbg << "SDL_CreateRGBSurfaceFrom Error: " << SDL_GetError() << std::endl;
		return -4;	
	}

	auto start = std::chrono::system_clock::now();

	int j = 0;
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

		if (j < mb.height())
		{
			j = draw_lines(mb, cam, *(world), rnd, j, 100, 20);
		}
		if (j == mb.height())
		{		
			auto end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_seconds = end - start;
			dbg << "Rendering took " << (elapsed_seconds.count() ) << " seconds" << std::endl;
			mb_save(mb, "test.png");
			j++;
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
