#ifndef __HITABLE_LIST_H__
#define __HITABLE_LIST_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "hitable.h"

class HitableList : public hitable {
	hitableptr * _list;
	int _count;
	int _max;

public:

	HitableList(int max) :
		hitable(LIST), 
		_max(max), 
		_count(0) {
		cudaMallocManaged(&_list, max * sizeof(hitable*));
	}

	~HitableList() {
		for (size_t i = 0; i < _count; i++)
		{
			delete _list[i];
		}
		cudaFree(_list);
	}

	__host__ bool add(hitableptr h) {
		if (_count == _max) {
			return false;
		}
		_list[_count++] = h;
		return true;
	}

	__device__ bool hit(const ray & r, float t_min, float t_max, hit_record & rec) const {
		hit_record temp_rec;
		bool hit_anything = false;
		float closest_so_far = t_max;		
		for (size_t i = 0; i < _count; i++)
		{
			if (_list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}
		return hit_anything;
	}
};


#endif
