#ifndef __CUDA_RENDER_H__
#define __CUDA_RENDER_H__

#include "irender.h"
#include "camera.h"
#include "../rt/memorybuffer.h"
#include "rnd.h"
#include "managed.h"

class hitable;

class BlockCalc : Managed {

	int _width;
	int _height;
	int _pixelsPerBlockX;
	int _pixelsPerBlockY;
	int _blocksX;
	int _blocksY;

	void updateBlocks() {
		_blocksX = _width / _pixelsPerBlockX + 1;
		_blocksY = _height / _pixelsPerBlockY + 1;
	}
public:

	__host__ BlockCalc() : _width(1), _height(1), _pixelsPerBlockX(1), _blocksX(1), _blocksY(1) {}
	
	__host__ void setBlockSize(int pixelsPerX, int pixelsPerY) {
		_pixelsPerBlockX = pixelsPerX;
		_pixelsPerBlockY = pixelsPerY;
		updateBlocks();
	}
	__host__ void setImage(int widthInPixels, int heightInPixels) {
		_width = widthInPixels;
		_height = heightInPixels;
		updateBlocks();
	}	

	__device__ __host__ void indexToXY(int blockIndex, int & x, int & y) const {
		y = blockIndex / _blocksX;
		x = blockIndex - (y * _blocksX);
	}
	
	__device__ __host__ void pixelOffset(int blockIndex, int & x, int & y) const {
		indexToXY(blockIndex, x, y);
		x *= _pixelsPerBlockX;
		y *= _pixelsPerBlockY;
	}

	__device__ __host__ int blockSum() const {
		return _blocksX * _blocksY;
	}
};


class CudaRender : public  IRender {	
	RGB8MemoryBuffer * _buffer;
	camera * _cam;
	hitable * _world;
	dim3 _blocks; 
	dim3 _threads;

	Rnd * _rnd;
	RGB8Color * _fbdev;
	int _currentBlock;
	BlockCalc _blockCalculator;
public:
	CudaRender(int blocksizeX=32, int blockSizeY=32, int blocksPerCompute=25);
	virtual std::string info();
	virtual int setup(RGB8MemoryBuffer * buffer, camera * cam);
	virtual int computeNext();
	virtual bool syncBuffers();
	virtual bool cleanup();
};



#endif
