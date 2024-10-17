#include <cuda_runtime.h>
#include <cuda_fp8.h>

template <typename T, int N>
struct GPUArray {
    T elements[N];  // Fixed-size array

    __device__ T& operator[](int index) {
        return elements[index];  // Access element by index
    }

    __device__ const T& operator[](int index) const {
        return elements[index];  // Access element by index (const version)
    }

    __device__ T& at(int index) {
        assert(index >= 0 && index < N);  // In device code, we use assert
        return elements[index];
    }

    __device__ const T& at(int index) const {
        assert(index >= 0 && index < N);
        return elements[index];
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

    __device__ void clear() {
        for (int i = 0; i < N; ++i) {
            elements[i] = static_cast<T>(0);  // Set each element to 0
        }
    }

};

class Bfloat16Wrapper {
    private:
        __nv_bfloat16 value; // The wrapped __nv_bfloat16 value
    public:
        __host__ __device__ Bfloat16Wrapper() : value(__float2bfloat16(0.0f)) {}
        __host__ __device__ Bfloat16Wrapper(float f) : value(__float2bfloat16(f)) {}
        __host__ __device__ operator float() const {
            return __bfloat162float(value);
        }
        __host__ __device__ Bfloat16Wrapper& operator=(float f) {
            value = __float2bfloat16(f);
            return *this;
        }
}; 

class HalfWrapper {
    private:
        __half value; 
    public:
        __host__ __device__ HalfWrapper() : value(__float2half(0.0f)) {}
        __host__ __device__ HalfWrapper(float f) : value(__float2half(f)) {}
        __host__ __device__ operator float() const {
            return __half2float(value);
        }
        __host__ __device__ HalfWrapper& operator=(float f) {
            value = __float2bfloat16(f);
            return *this;
        }
}; 


class FP8E5M2Wrapper {
private:
    __nv_fp8_e5m2 value;
public:
    __host__ __device__ FP8E5M2Wrapper() : value(__float2fp8_e5m2(0.0f)) {}
    __host__ __device__ FP8E5M2Wrapper(float f) : value(__float2fp8_e5m2(f)) {}
    __host__ __device__ operator float() const {
        return __fp8_e5m22float(value);
    }
    __host__ __device__ __nv_fp8_e5m2 raw() const {
        return value;
    }
};


class FP8E4M3Wrapper {
private:
    __nv_fp8_e4m3 value; 

public:
    __host__ __device__ FP8E4M3Wrapper() : value(__float2fp8_e4m3(0.0f)) {}
    __host__ __device__ FP8E4M3Wrapper(float f) : value(__float2fp8_e4m3(f)) {}
    __host__ __device__ operator float() const {
        return __fp8_e4m32float(value);
    }
    __host__ __device__ __nv_fp8_e4m3 raw() const {
        return value;
    }
};

