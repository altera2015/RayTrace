#ifndef __SCENE_H__
#define __SCENE_H__

class hitable;
class Rnd;

hitable * buildWorld(Rnd & rnd);
hitable *random_scene(Rnd &rnd);

#endif