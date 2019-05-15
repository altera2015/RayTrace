#include "dostream.h"
#include "sdlhelpers.h"
#include "memorybuffer.h"

#include "ray.h"
#include "sphere.h"
#include "material.h"
#include "hitable_list.h"
#include "camera.h"
#include "rnd.h"


static dostream dbg;


RGBColor color(const ray & r, const hitable & world, int depth )
{
	hit_record rec;
	if ( world.hit(r, 0.0001f, 10000.0f, rec))	
	{
		ray scattered;
		RGBColor attenuated;
		if (depth >= 0 && rec.mat_ptr->scatter(r, rec, attenuated, scattered))
		{
			return attenuated * color(scattered, world, depth - 1);
		}
		else
		{
			return RGBColor();
		}
	}

	vec3 dir = unit_vector(r.direction());
	float t = 0.5f * (dir.y() + 1.0f);
	return (1.0f - t) * RGBColor(1.0f, 1.0f, 1.0f) + t * RGBColor(0.5f, 0.7f, 1.0f);
}

void draw_line(RGB8MemoryBuffer & mb, camera & cam, const hitable & world, Rnd & rnd, int j, int samples = 100, int max_bounces = 50)
{		
	// for (int j = 0; j < mb.height() ; j++)
	{
		for (int i = 0; i < mb.width(); i++)
		{
			RGBColor col;
			for (int sample = 0; sample < samples; sample++)
			{
				float u = float(i + rnd.random() ) / float(mb.width());
				float v = float(j + rnd.random() ) / float(mb.height());

				ray r = cam.get_ray(u, v);
				col += color(r, world, max_bounces);
			}
			col *= (1.0f / samples);
			mb.put(i,j, RGB8Color(uint8_t(sqrt(col.r) * 255.99), uint8_t(sqrt(col.g) * 255.99), uint8_t(sqrt(col.b) * 255.99)));
		}
	}
}



hitable *random_scene(Rnd &rnd) {

	HitableList * hl = new HitableList();

	int n = 500;
	
	hl->add(new sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f, MaterialSharedPtr(new lambertian(RGBColor (0.5f, 0.5f, 0.5f), rnd))));

	int i = 1;
	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			float choose_mat = rnd.random();
			vec3 center(a + 0.9f*rnd.random(), 0.2f, b + 0.9f*rnd.random());
			if ((center - vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
				if (choose_mat < 0.8f) {  // diffuse
					hl->add(new sphere(center, 0.2f, MaterialSharedPtr(new lambertian(RGBColor(rnd.random()*rnd.random(), rnd.random()*rnd.random(), rnd.random()*rnd.random()), rnd))));
				}
				else if (choose_mat < 0.95) { // metal
					hl->add( new sphere(center, 0.2f, MaterialSharedPtr(new metal(RGBColor(0.5f*(1.0f + rnd.random()), 0.5f*(1.0f + rnd.random()), 0.5f*(1.0f + rnd.random())), 0.5f*rnd.random(),rnd))));
				}
				else {  // glass
					hl->add( new sphere(center, 0.2f, MaterialSharedPtr(new dielectric(1.5f, rnd))));
				}
			}
		}
	}

	hl->add(new sphere(vec3(0.0f, 1.0f, 0.0f), 1.0f, MaterialSharedPtr(new dielectric(1.5f, rnd))));
	hl->add(new sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, MaterialSharedPtr(new lambertian(RGBColor(0.4f, 0.2f, 0.1f), rnd))));
	hl->add(new sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, MaterialSharedPtr(new metal(RGBColor(0.7f, 0.6f, 0.5f), 0.0f, rnd))));


	return hl;
}

hitable * buildWorld(Rnd & rnd) {

	MaterialSharedPtr lamb_1(new lambertian(RGBColor(0.8f, 0.3f, 0.3f), rnd));
	MaterialSharedPtr lamb_2(new lambertian(RGBColor(0.8f, 0.8f, 0.0f), rnd));	
	MaterialSharedPtr metal_1(new metal(RGBColor(0.8f, 0.6f, 0.2f), 0.3f, rnd));
	MaterialSharedPtr metal_2(new metal(RGBColor(0.8f, 0.8f, 0.8f), 1.0f, rnd ));
	MaterialSharedPtr glass(new dielectric(1.5, rnd));
	

	HitableList * hl = new HitableList();
	hl->add(new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5, lamb_1));
	hl->add(new sphere(vec3(0.0f, -100.5f, -1.0f), 100, lamb_2));
	hl->add(new sphere(vec3(1.0f, 0.0f, -1.0f), 0.5, metal_1));
	hl->add(new sphere(vec3(-1.0f, 0.0f, -1.0f), 0.5, glass));

	return hl;
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
			draw_line(mb, cam, *(world), rnd, j, 100, 20);
			j++;
		}
		if (j == mb.height())
		{
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
