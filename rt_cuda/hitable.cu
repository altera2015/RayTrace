#include "hitable.h"
#include "hitable_list.h"
#include "sphere.h"

__device__ bool hitable::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
	// I am aware polymorphism exists
	// unfortnuately Cuda doesn't work with that across 
	// host / dev boundary.
	switch (type)
	{
	case LIST:
		return (static_cast<const HitableList*>(this))->hit(r, t_min, t_max, rec);
	case SPHERE:
		return (static_cast<const sphere*>(this))->hit(r, t_min, t_max, rec);
	}
	return false;
}


