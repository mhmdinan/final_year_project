import numpy as np
import time

N = 4096
if __name__ == "__main__":
    A = np.random.randn(N, N)
    B = np.random.randn(N, N)
    st = time.monotonic()
    C = A @ B
    et = time.monotonic()
    s = et-st
flop = N * N * 2 * N
print(f"{flop / 1e9:.2f} GFLOP")
print(f"{flop/s * 1e-9} GFLOP/s")