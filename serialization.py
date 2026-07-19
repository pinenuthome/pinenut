"""State-dictionary serialization."""

import numpy as np


def save(state_dict, path):
    if not isinstance(state_dict, dict):
        raise TypeError('save expects a state dictionary')
    np.savez(path, **state_dict)


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}
