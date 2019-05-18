#ifndef __MANAGED_H__
#define __MANAGED_H__

#include <cuda_runtime.h>


class Managed {
public:
	void *operator new(size_t len) {
		void *ptr;
		cudaError_t e = cudaMallocManaged(&ptr, len);
		if (e) {
			cudaDeviceReset();
			exit(99);
		}
		e = cudaDeviceSynchronize();
		if (e) {
			cudaDeviceReset();
			exit(99);
		}
		return ptr;
	}

	void operator delete(void *ptr) {
		cudaDeviceSynchronize();
		cudaError_t e = cudaFree(ptr);
		if (e) {
			exit(99);
		}
	}
};

#endif