#include <cstdio>
#include <cstring>
#include "acl/acl.h"
int main(int argc, char** argv) {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtContext ctx;
    aclrtCreateContext(&ctx, 0);
    uint32_t mid;
    aclError ret = aclmdlLoadFromFile(argv[1], &mid);
    printf("load ret=%u\n", ret);
    aclmdlDesc* desc = aclmdlCreateDesc();
    ret = aclmdlGetDesc(desc, mid);
    printf("desc ret=%d\n", (int)ret);
    size_t n_in = aclmdlGetNumInputs(desc), n_out = aclmdlGetNumOutputs(desc);
    printf("inputs=%zu outputs=%zu\n", n_in, n_out);
    aclmdlDataset* in_ds = aclmdlCreateDataset();
    void* dev_in;
    size_t in_size = aclmdlGetInputSizeByIndex(desc, 0);
    aclrtMalloc(&dev_in, in_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    void* host_in; aclrtMallocHost(&host_in, in_size);
    FILE* f = fopen("/work/input_1x3x640x640.bin", "rb");
    fread(host_in, 1, in_size, f); fclose(f);
    aclrtMemcpy(dev_in, in_size, host_in, in_size, ACL_MEMCPY_HOST_TO_DEVICE);
    aclDataBuffer* in_buf = aclCreateDataBuffer(dev_in, in_size);
    aclmdlAddDatasetBuffer(in_ds, in_buf);
    aclmdlDataset* out_ds = aclmdlCreateDataset();
    void* dev_outs[8]; size_t out_sizes[8];
    for (size_t i = 0; i < n_out; i++) {
        out_sizes[i] = aclmdlGetOutputSizeByIndex(desc, i);
        aclrtMalloc(&dev_outs[i], out_sizes[i], ACL_MEM_MALLOC_NORMAL_ONLY);
        aclDataBuffer* b = aclCreateDataBuffer(dev_outs[i], out_sizes[i]);
        aclmdlAddDatasetBuffer(out_ds, b);
        printf("out[%zu] size=%zu\n", i, out_sizes[i]);
    }
    printf("executing...\n"); fflush(stdout);
    ret = aclmdlExecute(mid, in_ds, out_ds);
    printf("execute ret=%d\n", (int)ret);
    if (ret == 0) {
        void* host_out; aclrtMallocHost(&host_out, out_sizes[0]);
        aclrtMemcpy(host_out, out_sizes[0], dev_outs[0], out_sizes[0], ACL_MEMCPY_DEVICE_TO_HOST);
        float* d = (float*)host_out;
        int cnt = 0;
        for (int i = 0; i < 300; i++)
            if (d[i*6+4] > 0.5f) { printf("  [%d] box=(%.3f,%.3f,%.3f,%.3f) conf=%.3f cls=%.0f\n", i, d[i*6], d[i*6+1], d[i*6+2], d[i*6+3], d[i*6+4], d[i*6+5]); cnt++; }
        printf("dets(conf>0.5): %d\n", cnt);
    }
    return 0;
}
