#ifndef __HITABLE_LIST_H__
#define __HITABLE_LIST_H__

#include <list>
#include "hitable.h"

class HitableList : public hitable {
	typedef std::list<HitablePtr> HistablePtrList;
	HistablePtrList _list;
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
};


#endif
