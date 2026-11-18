#include <cstdio>
#include "acl/acl.h"
int main(int argc, char** argv) {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtContext ctx;
    aclrtCreateContext(&ctx, 0);
    uint32_t mid;
    aclmdlLoadFromFile(argv[1], &mid);
    aclmdlDesc* desc = aclmdlCreateDesc();
    aclmdlGetDesc(desc, mid);
    printf("inputs=%u outputs=%u\n", aclmdlGetNumInputs(desc), aclmdlGetNumOutputs(desc));
    for (size_t i = 0; i < aclmdlGetNumInputs(desc); i++)
        printf("in[%zu] %s size=%zu\n", i, aclmdlGetInputNameByIndex(desc, i), aclmdlGetInputSizeByIndex(desc, i));
    for (size_t i = 0; i < aclmdlGetNumOutputs(desc); i++)
        printf("out[%zu] %s size=%zu\n", i, aclmdlGetOutputNameByIndex(desc, i), aclmdlGetOutputSizeByIndex(desc, i));
    return 0;
}
