#ifndef __HITABLE_LIST_H__
#define __HITABLE_LIST_H__

#include <vector>
#include <algorithm>
#include "hitable.h"
#include "dostream.h"

class HitableList : public hitable {
	typedef std::vector<HitablePtr> HistablePtrList;
	HistablePtrList _list;

	static int box_x_compare(const HitablePtr & a, const HitablePtr & b) {
		aabb box_left, box_right;
		
		if (!a->bounding_box(box_left) || !b->bounding_box(box_right))
			std::cerr << "no bounding box in bvh_node constructor\n";
		if (box_left.minvec().x() - box_right.minvec().x() < 0.0)
			return -1;
		else
			return 1;
	}

	static int box_y_compare(const HitablePtr & a, const HitablePtr & b)
	{
		aabb box_left, box_right;

		if (!a->bounding_box(box_left) || !b->bounding_box(box_right))
			std::cerr << "no bounding box in bvh_node constructor\n";
		if (box_left.minvec().y() - box_right.minvec().y() < 0.0)
			return -1;
		else
			return 1;
	}

	static int box_z_compare(const HitablePtr & a, const HitablePtr & b)
	{
		aabb box_left, box_right;

		if (!a->bounding_box(box_left) || !b->bounding_box(box_right))
			std::cerr << "no bounding box in bvh_node constructor\n";
		if (box_left.minvec().z() - box_right.minvec().z() < 0.0)
			return -1;
		else
			return 1;
	}



public:
	HitableList() {		
	}
	void add(hitable * h) {
		_list.push_back(HitablePtr(h));
	}

	bool hit(const ray & r, float t_min, float t_max, hit_record & rec) const {
		hit_record temp_rec;
		bool hit_anything = false;
		float closest_so_far = t_max;
		for (HistablePtrList::const_iterator i = _list.cbegin(); i != _list.cend(); i++)
		{
			if ( (*i)->hit(r, t_min, closest_so_far, temp_rec) ) {
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}
		return hit_anything;
	}

	bool bounding_box(aabb & box) const {
		if (_list.size() < 1) {
			return false;
		}

		bool first = true;
		aabb temp_box;		
		for (HistablePtrList::const_iterator i = _list.cbegin(); i != _list.cend(); i++)
		{
			bool ok = (*i)->bounding_box(temp_box);
			if (!ok) {
				return false;
			}
			if (first) {
				box = temp_box;
				first = false;
			}
			else 
			{
				box = aabb::surrounding_box(box, temp_box);
			}
		}

		return true;
	}

	void sort(int axis, int start, int count) {

		HistablePtrList::iterator first = std::next(_list.begin(), start);
		HistablePtrList::iterator last = std::next(first, count);

		switch (axis) {
		case 0:			
			std::sort(first, last, box_x_compare);
			break;
		case 1:
			std::sort(first, last, box_y_compare);
			break;
		case 2:
			std::sort(first, last, box_z_compare);
			break;
		default:
			{
				dostream dbg;
				dbg << "hitable_list::sort invalid axis" << std::endl;
			}
		}
		
	}

	hitable * operator[](int i)
	{
		if (i < _list.size())
		{
			return _list[i].get();
			// HistablePtrList::iterator i2 = std::next(_list.begin(), i);
			// return (*i2).get();
		}
		else
		{
			return nullptr;
		}		
	}
	size_t size() const {
		return _list.size();
	}

};


#endif
