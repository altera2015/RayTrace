#ifndef __COLOR_H__
#define __COLOR_H__

#include <stdint.h>

template<typename T>
struct TColor {
	T r;
	T g;
	T b;

	TColor(T pr = T(0), T pg = T(0), T pb = T(0)) : r(pr), g(pg), b(pb) {
	}
	
	inline TColor& operator*=(const float t) {
		r *= t;
		g *= t;
		b *= t;
		return *this;
	}
	inline TColor& operator*=(const double t) {
		r *= t;
		g *= t;
		b *= t;
		return *this;
	}

	inline TColor& operator*=(const TColor & other) {
		r *= other.r;
		g *= other.g;
		b *= other.b;
		return *this;
	}

	inline TColor& operator+=(const TColor & other ) {
		r += other.r;
		g += other.g;
		b += other.b;
		return *this;
	}
};

template<typename T>
inline TColor<T> operator*(float t, const TColor<T> &v) {
	return TColor<T>(t*v.r, t*v.g, t*v.b);
}

template<typename T>
inline TColor<T> operator*(const TColor<T> &v1, const TColor<T> &v2) {
	return TColor<T>(v1.r*v2.r, v1.g*v2.g, v1.b*v2.b);
}

template<typename T>
inline TColor<T> operator+(const TColor<T> &v1, const TColor<T> &v2) {
	return TColor<T>(v1.r + v2.r, v1.g + v2.g, v1.b + v2.b);
}



typedef TColor<uint8_t> RGB8Color;
typedef TColor<float> RGBColor;


#endif


