#include "sdlhelpers.h"
#include "dostream.h"

static dostream dbg;


SDL_SurfacePtr SDLH_SurfaceFromMemoryBuffer(RGB8MemoryBuffer & mb)
{
	uint32_t rmask = 0x000000ff;
	uint32_t gmask = 0x0000ff00;
	uint32_t bmask = 0x00ff0000;

	return SDL_SurfacePtr(SDL_CreateRGBSurfaceFrom(mb.raw(), mb.width(), mb.height(), mb.bytesPerPixel() * 8, mb.width() * mb.bytesPerPixel(), rmask, gmask, bmask, 0) );
}


bool SDLH_SetupWindow(int left, int top, int width, int height, SDL_WindowPtr & win, SDL_RendederPtr & ren)
{
	win.reset(SDL_CreateWindow("Render", left, top, width, height, SDL_WINDOW_SHOWN));
	if (win == nullptr) {
		dbg << "SDL_CreateWindow Error: " << SDL_GetError() << std::endl;
		return false;
	}
	
	ren.reset(SDL_CreateRenderer(win.get(), -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC));
	if (ren == nullptr) {
		dbg << "SDL_CreateRenderer Error: " << SDL_GetError() << std::endl;
		return false;
	}

	return true;
}