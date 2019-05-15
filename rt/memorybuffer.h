#ifndef __MEMORYBUFFER_H__
#define __MEMORYBUFFER_H__

#include "color.h"
#include <vector>
#include <fstream>


template<typename T>
class MemoryBuffer {

	int _width;
	int _height;
	std::vector<T> _memory;

public:
	MemoryBuffer(int width, int height, T def) :
		_width(width),
		_height(height)
	{
		_memory.resize(width * height * sizeof(T), def);
	}

	int width() const {
		return _width;
	}

	int height() const {
		return _height;
	}

	inline void put(int x, int y, T c)
	{
		_memory[y * _width + x] = c;
	}

	int bytesPerPixel() const {
		return sizeof(T);
	}

	size_t length() const {
		return _memory.size();
	}

	T * raw() {
		return _memory.data();
	}
	
};

typedef MemoryBuffer<RGB8Color> RGB8MemoryBuffer;
void mb_save(RGB8MemoryBuffer & buffer, const std::string & filename);

#endif
