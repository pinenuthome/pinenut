import numpy as np
import pinenut.core as pn
_cuda_available = True
try:
    import cupy as cp
    import cupyx as cpx
except ImportError:
    _cuda_available = False


class Cuda:
    @staticmethod
    def available():
        return _cuda_available

    @staticmethod
    def cupy():
        return cp

    @staticmethod
    def cupyx():
        return cpx

    @staticmethod
    def numpy():
        return np

    @staticmethod
    def xp():
        if _cuda_available:
            return cp
        else:
            return np

    @staticmethod
    def array_types():
        if _cuda_available:
            return [np.ndarray, cp.ndarray]
        else:
            return [np.ndarray]

    @staticmethod
    def get_array_module(x):
        if isinstance(x, pn.Tensor):
            x = x.data

        if _cuda_available:
            if isinstance(x, cp.ndarray):
                return cp
            else:
                return np
        else:
            return np

    @staticmethod
    def as_numpy(x):
        if isinstance(x, pn.Tensor):
            x = x.data

        if not _cuda_available:
            if np.isscalar(x):
                return np.array(x)
            else:
                return x
        else:
            return cp.asnumpy(x)

    @staticmethod
    def as_cupy(x):
        if isinstance(x, pn.Tensor):
            x = x.data

        if _cuda_available:
            if isinstance(x, cp.ndarray):
                return x
            else:
                return cp.array(x)
        else:
            raise RuntimeError('Cupy is not available.')
