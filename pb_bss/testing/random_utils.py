import numpy as np
from functools import wraps


def _force_correct_shape(f):
    """ This decorator sets the seed and fix the snr.

    :param f: Function to be wrapped
    :return: noise_signal
    """
    @wraps(f)
    def wrapper(*shape, **kwargs):
        if not shape:
            shape = (1,)
        elif isinstance(shape[0], (tuple, list)):
            shape = shape[0]

        return f(*shape, **kwargs)

    return wrapper


@_force_correct_shape
def uniform(*shape, data_type=np.complex128):

    def _uniform(data_type_local):
        return np.random.uniform(-1, 1, shape).astype(data_type_local)

    if data_type in (np.float32, np.float64):
        return _uniform(data_type)
    elif data_type is np.complex64:
        return _uniform(np.float32) + 1j * _uniform(np.float32)
    elif data_type is np.complex128:
        return _uniform(np.float64) + 1j * _uniform(np.float64)


@_force_correct_shape
def randn(*shape, dtype=np.complex128):

    def _randn(data_type_local):
        return np.random.randn(*shape).astype(data_type_local)

    if dtype in (np.float32, np.float64):
        return _randn(dtype)
    elif dtype is np.complex64:
        return _randn(np.float32) + 1j * _randn(np.float32)
    elif dtype is np.complex128:
        return _randn(np.float64) + 1j * _randn(np.float64)


def normal(*shape, dtype=np.complex128):
    return randn(*shape, dtype=dtype)


@_force_correct_shape
def hermitian(*shape, data_type=np.complex128):
    """ Assures a random positive-semidefinite hermitian matrix.

    :param shape:
    :param data_type:
    :return:
    """
    assert shape[-1] == shape[-2]
    matrix = uniform(shape, data_type)
    matrix = matrix + np.swapaxes(matrix, -1, -2).conj()
    np.testing.assert_allclose(matrix, np.swapaxes(matrix, -1, -2).conj())
    return matrix


@_force_correct_shape
def pos_def_hermitian(*shape, data_type=np.complex128):
    """ Assures a random POSITIVE-DEFINITE hermitian matrix.

    TODO: Can this be changed? Why do we need 2?

    :param shape:
    :param data_type:
    :return:
    """
    matrix = hermitian(*shape, data_type=data_type)
    matrix += np.broadcast_to(shape[-1] * 2 * np.eye(shape[-1]), shape)
    return matrix
