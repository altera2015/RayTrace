#include "scenes.h"


#include "hitable.h"
#include "material.h"
#include "rnd.h"
#include "sphere.h"
#include "hitable_list.h"

hitable *random_scene(Rnd &rnd) {

	HitableList * hl = new HitableList();

	int n = 500;

	hl->add(new sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f, MaterialSharedPtr(new lambertian(RGBColor(0.5f, 0.5f, 0.5f), rnd))));

	int i = 1;
	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			float choose_mat = rnd.random();
			vec3 center(a + 0.9f*rnd.random(), 0.2f, b + 0.9f*rnd.random());
			if ((center - vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
				if (choose_mat < 0.8f) {  // diffuse
					hl->add(new sphere(center, 0.2f, MaterialSharedPtr(new lambertian(RGBColor(rnd.random()*rnd.random(), rnd.random()*rnd.random(), rnd.random()*rnd.random()), rnd))));
				}
				else if (choose_mat < 0.95) { // metal
					hl->add(new sphere(center, 0.2f, MaterialSharedPtr(new metal(RGBColor(0.5f*(1.0f + rnd.random()), 0.5f*(1.0f + rnd.random()), 0.5f*(1.0f + rnd.random())), 0.5f*rnd.random(), rnd))));
				}
				else {  // glass
					hl->add(new sphere(center, 0.2f, MaterialSharedPtr(new dielectric(1.5f, rnd))));
				}
			}
		}
	}

	hl->add(new sphere(vec3(0.0f, 1.0f, 0.0f), 1.0f, MaterialSharedPtr(new dielectric(1.5f, rnd))));
	hl->add(new sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, MaterialSharedPtr(new lambertian(RGBColor(0.4f, 0.2f, 0.1f), rnd))));
	hl->add(new sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, MaterialSharedPtr(new metal(RGBColor(0.7f, 0.6f, 0.5f), 0.0f, rnd))));


	return hl;
}

hitable * buildWorld(Rnd & rnd) {

	MaterialSharedPtr lamb_1(new lambertian(RGBColor(0.8f, 0.3f, 0.3f), rnd));
	MaterialSharedPtr lamb_2(new lambertian(RGBColor(0.8f, 0.8f, 0.0f), rnd));
	MaterialSharedPtr metal_1(new metal(RGBColor(0.8f, 0.6f, 0.2f), 0.3f, rnd));
	MaterialSharedPtr metal_2(new metal(RGBColor(0.8f, 0.8f, 0.8f), 1.0f, rnd));
	MaterialSharedPtr glass(new dielectric(1.5, rnd));


	HitableList * hl = new HitableList();
	hl->add(new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5, lamb_1));
	hl->add(new sphere(vec3(0.0f, -100.5f, -1.0f), 100, lamb_2));
	hl->add(new sphere(vec3(1.0f, 0.0f, -1.0f), 0.5, metal_1));
	hl->add(new sphere(vec3(-1.0f, 0.0f, -1.0f), 0.5, glass));

	return hl;
}
