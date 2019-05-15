#ifndef __RND_H__
#define __RND_H__

#include <random>

class Rnd {

	std::random_device _rd;
	std::mt19937 _mt;
	std::uniform_real_distribution<float> _drand48;

public:
	Rnd() :
		_mt(_rd()),
		_drand48(0.0f, 1.0f)
	{
	}

	float random() {
		return _drand48(_mt);
	}

	vec3 random_in_unit_disk() {
		vec3 p;
		do {
			p = 2.0*vec3(_drand48(_mt), _drand48(_mt), 0) - vec3(1, 1, 0);
		} while (dot(p, p) >= 1.0);
		return p;
	}

	vec3 random_in_unit_sphere() {
		vec3 p;
		do {
			p = 2.0*vec3(_drand48(_mt), _drand48(_mt), _drand48(_mt)) - vec3(1, 1, 1);
		} while (p.squared_length() >= 1.0);
		return p;
	}
};

#endif
