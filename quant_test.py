import torch, torch_npu
out = open('/tmp/quant_test.txt', 'w')
out.write('start\n')
out.flush()
x = torch.randn(4, 4).npu().to(torch.float8_e5m2)
y = torch.randn(4, 4).npu().to(torch.float8_e5m2)
try:
    z = torch.mm(x, y)
    out.write('fp8 e5m2 matmul: OK\n')
except Exception as e:
    out.write(f'fp8 e5m2 matmul: FAIL {str(e)[:80]}\n')
out.flush()
xi = torch.randint(-127, 127, (4,4)).npu().to(torch.int8)
yi = torch.randint(-127, 127, (4,4)).npu().to(torch.int8)
try:
    zi = torch.mm(xi, yi)
    out.write('int8 matmul: OK\n')
except Exception as e:
    out.write(f'int8 matmul: FAIL {str(e)[:80]}\n')
out.write('done\n')
out.close()
