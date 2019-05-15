#ifndef __SDL_HELPERS_H__
#define __SDL_HELPERS_H__

#include <memory>
#include <SDL.h>
#include "memorybuffer.h"

struct WindowDeleter {
	void operator()(SDL_Window*w) {
		SDL_DestroyWindow(w);
	}
};


struct RenderDeleter {
	void operator()(SDL_Renderer*r) {
		SDL_DestroyRenderer(r);
	}
};

struct SurfaceDeleter {
	void operator()(SDL_Surface*s) {
		SDL_FreeSurface(s);
	}
};


struct TextureDeleter {
	void operator()(SDL_Texture*t) {
		SDL_DestroyTexture(t);
	}
};

typedef std::unique_ptr<SDL_Window, WindowDeleter > SDL_WindowPtr;
typedef std::unique_ptr<SDL_Renderer, RenderDeleter > SDL_RendederPtr;
typedef std::unique_ptr<SDL_Surface, SurfaceDeleter> SDL_SurfacePtr;
typedef std::unique_ptr<SDL_Texture, TextureDeleter> SDL_TexturePtr;

SDL_SurfacePtr SDLH_SurfaceFromMemoryBuffer(RGB8MemoryBuffer & mb);
bool SDLH_SetupWindow(int left, int top, int width, int height, SDL_WindowPtr & win, SDL_RendederPtr & ren);

#endif
