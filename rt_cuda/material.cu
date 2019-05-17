#include "material.h"

__device__ float schlick(float cosine, float ref_idx) {
	float r0 = (1 - ref_idx) / (1 + ref_idx);
	r0 = r0 * r0;
	return r0 + (1 - r0)*pow((1 - cosine), 5);
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
	vec3 uv = unit_vector(v);
	float dt = dot(uv, n);
	float discriminant = 1.0f - ni_over_nt * ni_over_nt*(1.0f - dt * dt);
	if (discriminant > 0.0f) {
		refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
		return true;
	}
	else
		return false;
}


__device__ vec3 reflect(const vec3& v, const vec3& n) {
	return v - 2 * dot(v, n)*n;
}

__device__ bool material::scatter(const ray & r_in, const hit_record & rec, RGBColor & attenuation, ray & scattered, Rnd & rnd)
{	
	switch (_type)
	{
	case LAMBERTIAN:
		return (static_cast<lambertian*>(this))->scatter(r_in, rec, attenuation, scattered, rnd);
	case METAL:
		return (static_cast<metal*>(this))->scatter(r_in, rec, attenuation, scattered, rnd);
	case DIELECTRIC:
		return (static_cast<dielectric*>(this))->scatter(r_in, rec, attenuation, scattered, rnd);

	}
	return false;
}
