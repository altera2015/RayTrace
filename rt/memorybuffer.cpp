#include "memorybuffer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "stb_image_write.h"

void mb_save(RGB8MemoryBuffer & buffer, const std::string & filename)
{
	stbi_write_png(filename.c_str(), buffer.width(), buffer.height(), 3, (void *)buffer.raw(), 0);
}
