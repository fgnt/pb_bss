import numpy as np


def uniform(*shape, data_type=np.complex128):
    if not shape:
        shape = (1,)
    elif isinstance(shape[0], tuple):
        shape = shape[0]

    def _uniform(data_type_local):
        return np.random.uniform(-1, 1, shape).astype(data_type_local)

    if data_type in (np.float32, np.float64):
        return _uniform(data_type)
    elif data_type is np.complex64:
        return _uniform(np.float32) + 1j * _uniform(np.float32)
    elif data_type is np.complex128:
        return _uniform(np.float64) + 1j * _uniform(np.float64)


def hermitian(*shape, data_type=np.complex128):
    assert shape[-1] == shape[-2]
    matrix = uniform(shape, data_type)
    matrix = matrix + np.swapaxes(matrix, -1, -2).conj()
    return matrix
