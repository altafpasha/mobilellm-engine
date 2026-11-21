#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <dlfcn.h>
#include <chrono>

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef size_t cl_size_t;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_mem;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_event;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU (1 << 2)

#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_WRITE_ONLY (1 << 1)
#define CL_MEM_COPY_HOST_PTR (1 << 5)

typedef cl_int (*pfn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*pfn_clGetDeviceIDs)(cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*);
typedef cl_context (*pfn_clCreateContext)(const void*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
typedef cl_command_queue (*pfn_clCreateCommandQueue)(cl_context, cl_device_id, cl_ulong, cl_int*);
typedef cl_mem (*pfn_clCreateBuffer)(cl_context, cl_ulong, size_t, void*, cl_int*);
typedef cl_program (*pfn_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (*pfn_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
typedef cl_kernel (*pfn_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_int (*pfn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (*pfn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*pfn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*pfn_clFinish)(cl_command_queue);
typedef cl_int (*pfn_clReleaseMemObject)(cl_mem);
typedef cl_int (*pfn_clReleaseKernel)(cl_kernel);
typedef cl_int (*pfn_clReleaseProgram)(cl_program);
typedef cl_int (*pfn_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (*pfn_clReleaseContext)(cl_context);

int main() {
    void* handle = dlopen("libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) handle = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) return 1;

#define LOAD_SYM(name) pfn_##name fn_##name = (pfn_##name)dlsym(handle, #name);
    LOAD_SYM(clGetPlatformIDs);
    LOAD_SYM(clGetDeviceIDs);
    LOAD_SYM(clCreateContext);
    LOAD_SYM(clCreateCommandQueue);
    LOAD_SYM(clCreateBuffer);
    LOAD_SYM(clCreateProgramWithSource);
    LOAD_SYM(clBuildProgram);
    LOAD_SYM(clCreateKernel);
    LOAD_SYM(clSetKernelArg);
    LOAD_SYM(clEnqueueNDRangeKernel);
    LOAD_SYM(clEnqueueReadBuffer);
    LOAD_SYM(clFinish);
    LOAD_SYM(clReleaseMemObject);
    LOAD_SYM(clReleaseKernel);
    LOAD_SYM(clReleaseProgram);
    LOAD_SYM(clReleaseCommandQueue);
    LOAD_SYM(clReleaseContext);

    cl_platform_id platform = nullptr;
    fn_clGetPlatformIDs(1, &platform, nullptr);
    cl_device_id device = nullptr;
    fn_clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_int err;
    cl_context ctx = fn_clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue queue = fn_clCreateCommandQueue(ctx, device, 0, &err);

    const uint32_t K = 2048;
    const uint32_t N = 2048;

    // Vectorized 128-bit loads (half8 = 16 bytes per memory bus transaction)
    const char* kernel_src =
        "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
        "__kernel void gemv_f16_f32_vec8(\n"
        "    __global const half8* weights, // [N, K/8]\n"
        "    __global const float* vec_in,  // [K]\n"
        "    __global float* vec_out,       // [N]\n"
        "    const uint K_over_8\n"
        ") {\n"
        "    uint row = get_global_id(0);\n"
        "    float sum0 = 0.0f, sum1 = 0.0f;\n"
        "    __global const half8* w_row = weights + row * K_over_8;\n"
        "    for (uint i = 0; i < K_over_8; i += 2) {\n"
        "        half8 w0 = w_row[i];\n"
        "        uint k0 = i * 8;\n"
        "        sum0 += (float)w0.s0 * vec_in[k0]   + (float)w0.s1 * vec_in[k0+1]\n"
        "              + (float)w0.s2 * vec_in[k0+2] + (float)w0.s3 * vec_in[k0+3]\n"
        "              + (float)w0.s4 * vec_in[k0+4] + (float)w0.s5 * vec_in[k0+5]\n"
        "              + (float)w0.s6 * vec_in[k0+6] + (float)w0.s7 * vec_in[k0+7];\n"
        "        half8 w1 = w_row[i + 1];\n"
        "        uint k1 = (i + 1) * 8;\n"
        "        sum1 += (float)w1.s0 * vec_in[k1]   + (float)w1.s1 * vec_in[k1+1]\n"
        "              + (float)w1.s2 * vec_in[k1+2] + (float)w1.s3 * vec_in[k1+3]\n"
        "              + (float)w1.s4 * vec_in[k1+4] + (float)w1.s5 * vec_in[k1+5]\n"
        "              + (float)w1.s6 * vec_in[k1+6] + (float)w1.s7 * vec_in[k1+7];\n"
        "    }\n"
        "    vec_out[row] = sum0 + sum1;\n"
        "}\n";

    cl_program prg = fn_clCreateProgramWithSource(ctx, 1, &kernel_src, nullptr, &err);
    err = fn_clBuildProgram(prg, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "FAIL: clBuildProgram failed: " << err << std::endl;
        return 2;
    }

    cl_kernel krn = fn_clCreateKernel(prg, "gemv_f16_f32_vec8", &err);

    std::vector<uint16_t> h_weights(N * K, 0x3c00); // 1.0 in FP16
    std::vector<float> h_in(K, 0.5f);
    std::vector<float> h_out(N, 0.0f);

    cl_mem d_weights = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * K * sizeof(uint16_t), h_weights.data(), &err);
    cl_mem d_in = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, K * sizeof(float), h_in.data(), &err);
    cl_mem d_out = fn_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, N * sizeof(float), nullptr, &err);

    uint32_t K_over_8 = K / 8;
    fn_clSetKernelArg(krn, 0, sizeof(cl_mem), &d_weights);
    fn_clSetKernelArg(krn, 1, sizeof(cl_mem), &d_in);
    fn_clSetKernelArg(krn, 2, sizeof(cl_mem), &d_out);
    fn_clSetKernelArg(krn, 3, sizeof(uint32_t), &K_over_8);

    size_t global_size = N;
    size_t local_size = 64; // Adreno 64-wide wave

    // Warmup
    for (int w = 0; w < 10; ++w) {
        fn_clEnqueueNDRangeKernel(queue, krn, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
    }
    fn_clFinish(queue);

    // Benchmark 100 iterations
    const int ITERS = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        fn_clEnqueueNDRangeKernel(queue, krn, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
    }
    fn_clFinish(queue);
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double per_layer_ms = (total_us / ITERS) / 1000.0;

    fn_clEnqueueReadBuffer(queue, d_out, 1, 0, N * sizeof(float), h_out.data(), 0, nullptr, nullptr);

    std::cout << "=== Coalesced 128-bit Vectorized OpenCL GEMV [2048 x 2048] ===" << std::endl;
    std::cout << "  Verification: output[0] = " << h_out[0] << " (expected 1024.0) -> " << (std::abs(h_out[0] - 1024.0f) < 1.0f ? "PASS" : "FAIL") << std::endl;
    std::cout << "  Per-GEMV Latency: " << per_layer_ms << " ms" << std::endl;

    double bytes_transferred = (double)N * K * sizeof(uint16_t) + K * sizeof(float) + N * sizeof(float);
    double effective_bandwidth_gb_s = (bytes_transferred / (per_layer_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "  Effective Memory Bandwidth: " << effective_bandwidth_gb_s << " GB/s" << std::endl;

    double projected_28l_decode_ms = per_layer_ms * 28 * 4;
    std::cout << "  Projected 28-layer 4-proj decode latency: " << projected_28l_decode_ms << " ms/token" << std::endl;
    std::cout << "  Projected decode throughput: " << (1000.0 / projected_28l_decode_ms) << " tok/s" << std::endl;

    fn_clReleaseMemObject(d_weights);
    fn_clReleaseMemObject(d_in);
    fn_clReleaseMemObject(d_out);
    fn_clReleaseKernel(krn);
    fn_clReleaseProgram(prg);
    fn_clReleaseCommandQueue(queue);
    fn_clReleaseContext(ctx);
    dlclose(handle);

    return 0;
}
