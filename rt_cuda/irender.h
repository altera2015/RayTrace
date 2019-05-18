#ifndef __RENDER_H__
#define __RENDER_H__

#include "../rt/memorybuffer.h"
#include "camera.h"


class IRender {
public:
	virtual int setup(RGB8MemoryBuffer * buffer, camera * cam) = 0;
	virtual int computeNext() = 0;
	virtual bool cleanup() = 0;
};

#endif
