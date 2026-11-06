#include <cstdio>
#include <cstdlib>
#include <chrono>
#include "acl/acl.h"

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 3000;
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtContext ctx;
    aclrtCreateContext(&ctx, 0);
    uint32_t model_id;
    if (aclmdlLoadFromFile("/work/yolo26n_cann901_bs1.om", &model_id) != 0) {
        printf("load fail\n"); return 1;
    }
    aclmdlDesc* md = aclmdlCreateDesc();
    aclmdlGetDesc(md, model_id);
    size_t in_sz = aclmdlGetInputSizeByIndex(md, 0);
    size_t out_sz = aclmdlGetOutputSizeByIndex(md, 0);
    void *dev_in, *dev_out;
    aclrtMalloc(&dev_in, in_sz, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&dev_out, out_sz, ACL_MEM_MALLOC_NORMAL_ONLY);
    float* h_in = (float*)malloc(in_sz);
    for (size_t i = 0; i < in_sz/4; i++) h_in[i] = 0.01f;
    aclrtMemcpy(dev_in, in_sz, h_in, in_sz, ACL_MEMCPY_HOST_TO_DEVICE);
    aclmdlDataset* ds_i = aclmdlCreateDataset();
    aclDataBuffer* bi = aclCreateDataBuffer(dev_in, in_sz);
    aclmdlAddDatasetBuffer(ds_i, bi);
    aclmdlDataset* ds_o = aclmdlCreateDataset();
    aclDataBuffer* bo = aclCreateDataBuffer(dev_out, out_sz);
    aclmdlAddDatasetBuffer(ds_o, bo);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    for (int i = 0; i < 20; i++) aclmdlExecuteAsync(model_id, ds_i, ds_o, stream);
    aclrtSynchronizeStream(stream);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        aclmdlExecuteAsync(model_id, ds_i, ds_o, stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    aclrtSynchronizeStream(stream);
    auto t2 = std::chrono::high_resolution_clock::now();
    double submit_us = std::chrono::duration<double, std::micro>(t1-t0).count() / N;
    double total_ms = std::chrono::duration<double, std::milli>(t2-t0).count() / N;
    printf("C++ ACL bs1: submit=%.0f us/img  total=%.3f ms/img\n", submit_us, total_ms);
    return 0;
}
