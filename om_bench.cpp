#include <cstdio>
#include <chrono>
#include "acl/acl.h"
int main(int argc, char** argv) {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtContext ctx; aclrtCreateContext(&ctx, 0);
    uint32_t mid;
    aclmdlLoadFromFile(argv[1], &mid);
    aclmdlDesc* desc = aclmdlCreateDesc();
    aclmdlGetDesc(desc, mid);
    size_t n_out = aclmdlGetNumOutputs(desc);
    aclmdlDataset* in_ds = aclmdlCreateDataset();
    size_t in_size = aclmdlGetInputSizeByIndex(desc, 0);
    void *dev_in, *h_in;
    aclrtMalloc(&dev_in, in_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMallocHost(&h_in, in_size);
    FILE* f = fopen("/work/input_1x3x640x640.bin", "rb");
    fread(h_in, 1, in_size, f); fclose(f);
    aclrtMemcpy(dev_in, in_size, h_in, in_size, ACL_MEMCPY_HOST_TO_DEVICE);
    aclmdlAddDatasetBuffer(in_ds, aclCreateDataBuffer(dev_in, in_size));
    aclmdlDataset* out_ds = aclmdlCreateDataset();
    void* dev_outs[8];
    for (size_t i = 0; i < n_out; i++) {
        size_t s = aclmdlGetOutputSizeByIndex(desc, i);
        aclrtMalloc(&dev_outs[i], s, ACL_MEM_MALLOC_NORMAL_ONLY);
        aclmdlAddDatasetBuffer(out_ds, aclCreateDataBuffer(dev_outs[i], s));
    }
    for (int i = 0; i < 10; i++) aclmdlExecute(mid, in_ds, out_ds);
    auto t0 = std::chrono::high_resolution_clock::now();
    int N = 50;
    for (int i = 0; i < N; i++) aclmdlExecute(mid, in_ds, out_ds);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1-t0).count() / N;
    printf("latency avg=%.2f ms -> %.1f fps\n", ms, 1000.0/ms);
    return 0;
}
