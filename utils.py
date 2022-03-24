import numpy as np

def rand_unitary_matrix(N0, tag_complex=True, seed=None):
    np_rng = np.random.default_rng(seed)
    if tag_complex:
        tmp0 = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0))
        tmp0 = tmp0 + tmp0.T.conj()
    else:
        tmp0 = np_rng.normal(size=(N0,N0))
        tmp0 = tmp0 + tmp0.T
    ret = np.linalg.eigh(tmp0)[1]
    return ret


def rand_hermite_matrix(N0, min_eig=1, max_eig=2, tag_complex=True, seed=None):
    np_rng = np.random.default_rng(seed)
    if (min_eig is None) and (max_eig is None):
        if tag_complex:
            tmp0 = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0))
            ret = tmp0 + tmp0.T.conj()
        else:
            tmp0 = np_rng.normal(size=(N0,N0))
            ret = tmp0 + tmp0.T
    else:
        assert (min_eig is not None) and (max_eig is not None)
        EVL = np_rng.uniform(min_eig, max_eig, size=(N0,))
        EVC = rand_unitary_matrix(N0, tag_complex, seed=np_rng.integers(10000))
        tmp0 = EVC.T.conj() if tag_complex else EVC.T
        ret = (EVC * EVL) @ tmp0
    return ret


