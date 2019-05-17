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

#ifndef HITABLEH
#define HITABLEH 

#include "ray.h"
#include "managed.h"
#include <memory>
#include <cuda.h>

class material;

struct hit_record
{
	float t;
	vec3 p;
	vec3 normal;
	material * mat_ptr;
};

enum HitableType { LIST, SPHERE };

class hitable : public Managed {	
	hitable(const hitable & other) {};//no copy!
public:
	HitableType type;
	__host__ virtual ~hitable() {}


	hitable(HitableType ptype) : type(ptype) {
	}

	__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
};

typedef hitable * hitableptr;


#endif
