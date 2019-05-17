#include "scenes.h"


#include "hitable.h"
#include "material.h"
#include "rnd.h"
#include "sphere.h"
#include "hitable_list.h"
#include <random>


hitable *random_scene() {

	std::random_device _rd;
	std::mt19937 _mt(_rd());
	std::uniform_real_distribution<float> _drand48(0.0f, 1.0f);

	auto random = [&_drand48, &_mt]() {
		return _drand48(_mt);
	};


	int n = 500;

	HitableList * hl = new HitableList(n + 10);
	

	hl->add(new sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new lambertian(RGBColor(0.5f, 0.5f, 0.5f))));

	int i = 1;
	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			float choose_mat = random();
			vec3 center(a + 0.9f*random(), 0.2f, b + 0.9f*random());
			if ((center - vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
				if (choose_mat < 0.8f) {  // diffuse
					hl->add(new sphere(center, 0.2f, new lambertian(RGBColor(random()*random(), random()*random(), random()*random()))));
				}
				else if (choose_mat < 0.95) { // metal
					hl->add(new sphere(center, 0.2f, new metal(RGBColor(0.5f*(1.0f + random()), 0.5f*(1.0f + random()), 0.5f*(1.0f + random())), 0.5f*random())));
				}
				else {  // glass
					hl->add(new sphere(center, 0.2f, new dielectric(1.5f)));
				}
			}
		}
	}

	hl->add(new sphere(vec3(0.0f, 1.0f, 0.0f), 1.0f, new dielectric(1.5f)));
	hl->add(new sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, new lambertian(RGBColor(0.4f, 0.2f, 0.1f))));
	hl->add(new sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, new metal(RGBColor(0.7f, 0.6f, 0.5f), 0.0f)));


	return hl;
}

hitable * buildWorld() {

	material * lamb_1(new lambertian(RGBColor(0.8f, 0.3f, 0.3f)));
	material *  lamb_2(new lambertian(RGBColor(0.8f, 0.8f, 0.0f)));
	material *  metal_1(new metal(RGBColor(0.8f, 0.6f, 0.2f), 0.3f));
	material *  metal_2(new metal(RGBColor(0.8f, 0.8f, 0.8f), 1.0f));
	material *  glass(new dielectric(1.5f));


	HitableList * hl = new HitableList(10);
	hl->add(new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5, lamb_1));
	hl->add(new sphere(vec3(0.0f, -100.5f, -1.0f), 100, lamb_2));
	hl->add(new sphere(vec3(1.0f, 0.0f, -1.0f), 0.5, metal_1));
	hl->add(new sphere(vec3(-1.0f, 0.0f, -1.0f), 0.5, glass));

	return hl;
}
