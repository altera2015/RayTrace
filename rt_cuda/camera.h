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

#ifndef CAMERAH
#define CAMERAH

#define _USE_MATH_DEFINES
#include <cmath>
#include "ray.h"
#include "rnd.h"

#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979f
#endif

class camera {
	Rnd _rnd;
public:

	__device__ __host__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist, Rnd rnd) : _rnd(rnd) { // vfov is top to bottom in degrees
		lens_radius = aperture / 2.0f;
		float theta = vfov * float(M_PI) / 180.0f;
		float half_height = tan(theta / 2);
		float half_width = aspect * half_height;
		origin = lookfrom;
		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);		
		upper_right_corner = origin - half_width*focus_dist*u + half_height*focus_dist*v - focus_dist*w;
		horizontal = 2 * half_width*focus_dist*u;
		vertical = -2 * half_height*focus_dist*v;
	}

	__device__ ray get_ray_pinhole(float s, float t) {
		vec3 offset(0.0f, 0.0f, 0.0f);
		return ray(origin + offset, upper_right_corner + s*horizontal + t*vertical - origin - offset);
	}
	__device__ ray get_ray(float s, float t) {
		vec3 rd = lens_radius*_rnd.random_in_unit_disk();
		vec3 offset = u * rd.x() + v * rd.y();
		return ray(origin + offset, upper_right_corner + s*horizontal + t*vertical - origin - offset);
	}

	vec3 origin;	
	vec3 upper_right_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	float lens_radius;
};
#endif
