#include <cuda_runtime.h>


template <typename T, int N>
struct GPUArray {
    T elements[N];  // Fixed-size array

    __device__ T& operator[](int index) {
        return elements[index];  // Access element by index
    }

    __device__ const T& operator[](int index) const {
        return elements[index];  // Access element by index (const version)
    }

    __device__ GPUArray<T, N> operator*(T scalar) const {
        GPUArray<T, N> result;
        for (int i = 0; i < N; ++i) {
            result[i] = elements[i] * scalar;  // Multiply each element by the scalar
        }
        return result;
    }

    __device__ GPUArray<T, N> operator+(T scalar) const {
        GPUArray<T, N> result;
        for (int i = 0; i < N; ++i) {
            result[i] = elements[i] + scalar;  // Add the scalar to each element
        }
        return result;
    }

    // In-place scalar multiplication (scaling)
    __device__ GPUArray<T, N>& operator*=(T scalar) {
        for (int i = 0; i < N; ++i) {
            elements[i] *= scalar;  // Multiply each element by the scalar
        }
        return *this;
    }

    // In-place scalar addition
    __device__ GPUArray<T, N>& operator+=(T scalar) {
        for (int i = 0; i < N; ++i) {
            elements[i] += scalar;  // Add the scalar to each element
        }
        return *this;
    }  

};

