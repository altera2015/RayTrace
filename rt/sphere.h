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

#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"
#include "material.h"

#include <stdint.h>
extern uint64_t hit_count;

class sphere : public hitable {
public:
	sphere() {}
	sphere(vec3 cen, float r, MaterialSharedPtr m) : center(cen), radius(r), mat_ptr(m) {};
	vec3 center;
	float radius;
	MaterialSharedPtr mat_ptr;


	bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
		hit_count++;
		vec3 oc = r.origin() - center;
		float a = dot(r.direction(), r.direction());
		float b = dot(oc, r.direction());
		float c = dot(oc, oc) - radius*radius;
		float discriminant = b*b - a*c;
		if (discriminant > 0) {
			float temp = (-b - sqrt(discriminant)) / a;
			if (temp < t_max && temp > t_min) {
				rec.t = temp;
				rec.p = r.point_at_parameter(rec.t);
				rec.normal = (rec.p - center) / radius;
				rec.mat_ptr = mat_ptr.get();
				return true;
			}
			temp = (-b + sqrt(discriminant)) / a;
			if (temp < t_max && temp > t_min) {
				rec.t = temp;
				rec.p = r.point_at_parameter(rec.t);
				rec.normal = (rec.p - center) / radius;
				rec.mat_ptr = mat_ptr.get();
				return true;
			}
		}
		return false;
	}

	virtual bool bounding_box(aabb & box) const {
		box = aabb(center - vec3(radius, radius, radius), center + vec3(radius, radius, radius));
		return true;
	}

};


#endif