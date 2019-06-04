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

#ifndef BVHH
#define BVHH

#include "hitable.h"
#include "hitable_list.h"
#include "dostream.h"
#include <list>
#include <string>
#include <stdint.h>
extern uint64_t hit_count;

class bvh_node : public hitable {

	HitableList * _list;
	

public:
	bvh_node() : _list(nullptr) {}

	~bvh_node() {
		delete _list;
	}
	
	void print(std::string parent, hitable * h) {
		dostream dbg;
		dbg << "\"" << parent << "\" -> ";
		dbg << "\"" << h << "\"" << std::endl;
	}
	void print(std::string parent, std::string self) {
		dostream dbg;
		if (parent.length() == 0) {
			return;
		}
		dbg << "\"" << parent << "\" -> ";
		dbg << "\"" << self << "\"" << std::endl;
	}

	bvh_node(HitableList * ol, int start = -1, int count = -1, std::string parent = std::string()) {
	
		dostream dbg;
		HitableList & l = *ol;
		if (start == -1) {
			_list = ol;
			start = 0;
			count = l.size();
		}
		else {
			_list = nullptr;
		}
		static int axis = -1; // int(3 * drand48());
		axis = ( axis + 1 ) % 3;
		// dbg << "axis " << axis << std::endl;
		l.sort(axis, start, count);
		
		char buf[100];
		sprintf(buf, "BVH %d", this);

		print(parent, buf);
		parent = buf;

		if (count == 1) {
			left = right = l[start + 0];
			print(parent, left);
			print(parent, right);
		}
		else if (count == 2) {
			left = l[start + 0];
			right = l[start + 1];
			print(parent, left);
			print(parent, right);
		}
		else {

			left = new bvh_node(ol, start, count / 2, parent);
			right = new bvh_node(ol, start + count / 2, count - count / 2, parent);
		}
		aabb box_left, box_right;
		if (!left->bounding_box(box_left) || !right->bounding_box(box_right))
			dbg  << "no bounding box in bvh_node constructor\n";
		box = aabb::surrounding_box(box_left, box_right);
		// dbg << box.volume() << " -- " << box._min << " " << box._max << std::endl;;
	}

	bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
		hit_count++;
		if (box.hit(r, t_min, t_max)) {
			hit_record left_rec, right_rec;
			bool hit_left = left->hit(r, t_min, t_max, left_rec);
			bool hit_right = right->hit(r, t_min, t_max, right_rec);
			if (hit_left && hit_right) {
				if (left_rec.t < right_rec.t)
					rec = left_rec;
				else
					rec = right_rec;
				return true;
			}
			else if (hit_left) {
				rec = left_rec;
				return true;
			}
			else if (hit_right) {
				rec = right_rec;
				return true;
			}
			else
				return false;
		}
		else return false;
	}


	bool bounding_box(aabb& b) const {
		b = box;
		return true;
	}
	hitable *left;
	hitable *right;
	aabb box;
};








#endif

