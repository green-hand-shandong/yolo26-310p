#include <cstdio>
#include <cstring>
#include "acl/acl.h"
int main(int argc, char** argv) {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtContext ctx; aclrtCreateContext(&ctx, 0);
    aclrtSetCurrentContext(ctx);
    uint32_t mid;
    aclError ret = aclmdlLoadFromFile(argv[1], &mid);
    printf("load ret=%u\n", ret); fflush(stdout);
    aclmdlDesc* desc = aclmdlCreateDesc();
    ret = aclmdlGetDesc(desc, mid);
    printf("desc ret=%d\n", (int)ret); fflush(stdout);
    size_t n_in = aclmdlGetNumInputs(desc), n_out = aclmdlGetNumOutputs(desc);
    printf("inputs=%zu outputs=%zu\n", n_in, n_out); fflush(stdout);
    aclmdlDataset* in_ds = aclmdlCreateDataset();
    size_t in_size = aclmdlGetInputSizeByIndex(desc, 0);
    void *dev_in, *h_in;
    aclrtMalloc(&dev_in, in_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMallocHost(&h_in, in_size);
    FILE* f = fopen("/work/input_obb_1024_real.bin", "rb");
    fread(h_in, 1, in_size, f); fclose(f);
    aclrtMemcpy(dev_in, in_size, h_in, in_size, ACL_MEMCPY_HOST_TO_DEVICE);
    aclmdlAddDatasetBuffer(in_ds, aclCreateDataBuffer(dev_in, in_size));
    aclmdlDataset* out_ds = aclmdlCreateDataset();
    void* dev_outs[4]; size_t out_sizes[4];
    for (size_t i = 0; i < n_out && i < 4; i++) {
        out_sizes[i] = aclmdlGetOutputSizeByIndex(desc, i);
        aclrtMalloc(&dev_outs[i], out_sizes[i], ACL_MEM_MALLOC_NORMAL_ONLY);
        aclmdlAddDatasetBuffer(out_ds, aclCreateDataBuffer(dev_outs[i], out_sizes[i]));
        printf("out[%zu] size=%zu\n", i, out_sizes[i]); fflush(stdout);
    }
    printf("executing...\n"); fflush(stdout);
    ret = aclmdlExecute(mid, in_ds, out_ds);
    printf("execute ret=%d\n", (int)ret); fflush(stdout);
    if (ret == 0) {
        void* h_out; aclrtMallocHost(&h_out, out_sizes[0]);
        aclrtMemcpy(h_out, out_sizes[0], dev_outs[0], out_sizes[0], ACL_MEMCPY_DEVICE_TO_HOST);
        float* d = (float*)h_out;  // (300, 7)
        int cnt = 0;
        for (int i = 0; i < 300; i++)
            if (d[i*7+4] > 0.25f) {
                printf("  [%d] box=(%.3f,%.3f,%.3f,%.3f) conf=%.3f cls=%.0f\n", i,
                       d[i*7], d[i*7+1], d[i*7+2], d[i*7+3], d[i*7+4], d[i*7+6]);
                cnt++;
            }
        printf("dets(conf>0.25): %d\n", cnt); fflush(stdout);
        // 保存供 python 对拍
        FILE* fo = fopen("/work/out_obb92_cpp.bin", "wb");
        fwrite(d, 4, 300*7, fo); fclose(fo);
    }
    return 0;
}
