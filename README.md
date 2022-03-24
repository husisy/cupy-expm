# cupy-expm

```Python
import cupy_extension
tmp0 = cp.random.rand(200, 200)
A = tmp0 + tmp0.T #make it symmetry matrix
ret = cupy_extension.expm(A)
```

```bash
$ python draft00.py
norm: 32.79106373312613
time(scipy): 1.9017131328582764
time(numpy_extension)  1.7555665969848633
time(cupy_extension)  1.0561015605926514
norm(diff(numpy_extension)): 5.363849691015013e-14
norm(diff(cupy_extension)): 2.025168132691914e-12
```

notice: `numpy_extension.py` is copied from [michael-hartmann/expm](https://github.com/michael-hartmann/expm)
