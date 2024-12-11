import numpy as np
import tvm
import tvm.testing
from tvm import te
from tvm.script.ir_builder import IRBuilder
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tir.transform import LowerTIRp
from tvm.contrib import cublas

@tvm.testing.requires_cuda_compute_version(8)
def test_agemm_ampere():
    Ms = [1024, 2048, 4096, 5376, 8192]
    Ns = [1024, 2048, 4096, 5376, 8192]
    Ks = [512, 1024, 2048, 4096, 8192]
    for M in Ms:
        for N in Ns:
            for K in Ks:
                if M != N:
                    continue
                # Matrix dimensions
                MI, NI, KI = 128, 128, 32
                MII, NII, KII = 64, 64, 16
                wmmaM, wmmaN = 16, 16
                warp_size = 32

                BLK_M, BLK_N, BLK_K = 128, 128, 32
                VEC = 8
                DEPTH = 2

                np.random.seed(0)
                A_np = np.random.randn(M, K).astype(np.float16)
                B_np = np.random.randn(N, K).astype(np.float16)

                DEV = tvm.cuda()
                A_tvm = tvm.nd.array(A_np, device=DEV)
                B_tvm = tvm.nd.array(B_np, device=DEV)

                target = tvm.target.Target.from_device(DEV)
                print(target)

                # cublas
                def cublas_gemm():
                    A = te.placeholder((M, K), name="A", dtype="float16")
                    B = te.placeholder((N, K), name="B", dtype="float16")
                    C = cublas.matmul(A, B, transb=True, dtype="float32")
                    s = te.create_schedule(C.op)

                    C_np = np.zeros((M, N), dtype=np.float32)
                    C_tvm = tvm.nd.array(C_np, device=DEV)
                    mod_cublaslt = tvm.build(s, [A, B, C], target)
                    mod_cublaslt(A_tvm, B_tvm, C_tvm)
                    timer = mod_cublaslt.time_evaluator(mod_cublaslt.entry_name, DEV, number=10, repeat=3)
                    res = timer(A_tvm, B_tvm, C_tvm)
                    print("cublas time: ")
                    print(res)

                    return C_tvm

                with target:
                    print(f"M, N, K: {M}, {N}, {K}")
                    C_cublas = cublas_gemm()
    
if __name__ == "__main__":
    test_agemm_ampere()