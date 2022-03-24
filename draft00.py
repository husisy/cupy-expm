import time
import numpy as np
import cupy as cp
import scipy.linalg

from utils import rand_hermite_matrix
import numpy_extension
import cupy_extension


N0 = 2000
A = rand_hermite_matrix(N0, min_eig=-1, max_eig=2, tag_complex=False)
print("norm ", np.linalg.norm(A, ord=1))

t0 = time.time()
ret_ = scipy.linalg.expm(A)
print("time(scipy):", time.time()-t0)

t0 = time.time()
ret_np  = numpy_extension.expm(A)
print("time(numpy_extension) ", time.time()-t0)

device = cp.cuda.Device(0)
device.use()
A_cp = cp.array(A)
device.synchronize()
t0 = time.time()
ret_cp = cupy_extension.expm(A_cp)
device.synchronize()
print("time(cupy_extension) ", time.time()-t0)

print("norm(diff(numpy_extension)):", np.linalg.norm(ret_np-ret_, ord=1))
print("norm(diff(cupy_extension)):", np.linalg.norm(ret_cp.get()-ret_, ord=1))
