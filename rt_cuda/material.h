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

#ifndef MATERIALH
#define MATERIALH 

struct hit_record;

#include <cuda.h>
#include "ray.h"
#include "hitable.h"
#include "rnd.h"
#include "color.h"
#include "managed.h"
#include <memory>


__device__ float schlick(float cosine, float ref_idx);
__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted);
__device__ vec3 reflect(const vec3& v, const vec3& n);

class material : public Managed {
public:	
	enum Type{LAMBERTIAN, METAL, DIELECTRIC};
protected:
	Type _type;
public:
	material(Type type) : _type(type) {};
	__device__ bool scatter(const ray& r_in, const hit_record& rec, RGBColor& attenuation, ray& scattered, Rnd & rnd);
};

class lambertian : public material {	
	RGBColor albedo;
public:

	lambertian(const RGBColor & a ) : material(LAMBERTIAN), albedo(a) {}
	__device__ bool scatter(const ray& r_in, const hit_record& rec, RGBColor& attenuation, ray& scattered, Rnd & rnd) {
		vec3 target = rec.p + rec.normal + rnd.random_in_unit_sphere();
		scattered = ray(rec.p, target - rec.p);
		attenuation = albedo;
		return true;
	}	
};

class metal : public material {
	
	RGBColor albedo;
	float fuzz;

public:



	__host__ metal(const RGBColor& a, float f ) : material(METAL), albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
	__device__ bool scatter(const ray& r_in, const hit_record& rec, RGBColor& attenuation, ray& scattered, Rnd & rnd) {
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		scattered = ray(rec.p, reflected + fuzz*rnd.random_in_unit_sphere());
		attenuation = albedo;
		return (dot(scattered.direction(), rec.normal) > 0);
	}
		
};

class dielectric : public material {
	
	float ref_idx;
public:

	
	dielectric(float ri ) : material(DIELECTRIC), ref_idx(ri) {}
	__device__ bool scatter(const ray& r_in, const hit_record& rec, RGBColor& attenuation, ray& scattered, Rnd & rnd) {
		vec3 outward_normal;
		vec3 reflected = reflect(r_in.direction(), rec.normal);
		float ni_over_nt;
		attenuation = RGBColor(1.0f, 1.0f, 1.0f);
		vec3 refracted;
		float reflect_prob;
		float cosine;
		if (dot(r_in.direction(), rec.normal) > 0) 
		{
			outward_normal = -rec.normal;
			ni_over_nt = ref_idx;
			// cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
			cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
			cosine = sqrt(1 - ref_idx*ref_idx*(1 - cosine*cosine));
		}
		else 
		{
			outward_normal = rec.normal;
			ni_over_nt = 1.0f / ref_idx;
			cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
		}
		if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
			reflect_prob = schlick(cosine, ref_idx);
		else
			reflect_prob = 1.0f;
		if (rnd.random() < reflect_prob)
		{	
			scattered = ray(rec.p, reflected);
		}
		else
		{
			scattered = ray(rec.p, refracted);
		}
		
		return true;
	}

	
};


#endif

