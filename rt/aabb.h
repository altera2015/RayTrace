//==================================================================================================
// Written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is distributed
// without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication along
// with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==================================================================================================

#ifndef AABBH
#define AABBH
#include "vec3.h"
#include "ray.h"
#include "hitable.h"

inline float ffmin(float a, float b) { return a < b ? a : b; }
inline float ffmax(float a, float b) { return a > b ? a : b; }

class aabb {

public:
	vec3 _min;
	vec3 _max;


	aabb() {}
	aabb(const vec3& a, const vec3& b) { _min = a; _max = b; }

	vec3 minvec() const { return _min; }
	vec3 maxvec() const { return _max; }

	bool hit(const ray& r, float tmin, float tmax) const {
		for (int a = 0; a < 3; a++) {
			float t0 = ffmin((_min[a] - r.origin()[a]) / r.direction()[a],
				(_max[a] - r.origin()[a]) / r.direction()[a]);
			float t1 = ffmax((_min[a] - r.origin()[a]) / r.direction()[a],
				(_max[a] - r.origin()[a]) / r.direction()[a]);
			tmin = ffmax(t0, tmin);
			tmax = ffmin(t1, tmax);
			if (tmax <= tmin)
				return false;
		}
		return true;
	}



	static aabb surrounding_box(aabb box0, aabb box1) {
		vec3 smallvec(fmin(box0.minvec().x(), box1.minvec().x()),
			fmin(box0.minvec().y(), box1.minvec().y()),
			fmin(box0.minvec().z(), box1.minvec().z()));
		vec3 big(fmax(box0.maxvec().x(), box1.maxvec().x()),
			fmax(box0.maxvec().y(), box1.maxvec().y()),
			fmax(box0.maxvec().z(), box1.maxvec().z()));
		return aabb(smallvec, big);
	}

	double volume() {
		vec3 v = _max - _min;
		return v[0] * v[1] * v[2];
	}
};


#endif
