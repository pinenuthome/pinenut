"""CUDA array helpers used by tensors, modules, and data loaders."""

import numpy as np


try:
    import cupy as cp
    import cupyx as cpx
except ImportError:
    cp = None
    cpx = None


def is_available():
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def device_count():
    return cp.cuda.runtime.getDeviceCount() if is_available() else 0


def cupy():
    if cp is None:
        raise RuntimeError('CuPy is not installed')
    return cp


def cupyx():
    if cpx is None:
        raise RuntimeError('CuPy is not installed')
    return cpx


def numpy():
    return np


def xp():
    return cp if is_available() else np


def array_types():
    return (np.ndarray, cp.ndarray) if cp is not None else (np.ndarray,)


def _unwrap(value):
    if hasattr(value, 'grad') and hasattr(value, 'creator'):
        return value.data
    return value


def get_array_module(value):
    value = _unwrap(value)
    if cp is not None and isinstance(value, cp.ndarray):
        return cp
    return np


def as_numpy(value):
    value = _unwrap(value)
    if cp is not None and isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    if np.isscalar(value):
        return np.asarray(value)
    return value


def as_cupy(value, device=None):
    if cp is None:
        raise RuntimeError('CuPy is not installed')
    value = _unwrap(value)
    device_id = _device_index(device)
    with cp.cuda.Device(device_id):
        return cp.asarray(value)


def _device_index(device):
    if isinstance(device, int):
        return device
    if isinstance(device, str) and device.startswith('cuda:'):
        return int(device.split(':', 1)[1])
    if device in (None, 'cuda'):
        return 0
    raise ValueError("device must be 'cpu', 'cuda', 'cuda:N', or an integer")
