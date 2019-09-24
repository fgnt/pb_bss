import numpy as np
from numpy.testing.utils import assert_array_compare, assert_array_less
import operator


def assert_array_greater(x, y, err_msg='', verbose=True):
    assert_array_compare(operator.__gt__, x, y, err_msg=err_msg,
                         verbose=verbose,
                         header='Arrays are not greater-ordered')


def assert_array_greater_equal(x, y, err_msg='', verbose=True):
    assert_array_compare(operator.__ge__, x, y, err_msg=err_msg,
                         verbose=verbose,
                         header='Arrays are not greater-ordered')


def assert_array_less_equal(x, y, err_msg='', verbose=True):
    assert_array_compare(operator.__le__, x, y, err_msg=err_msg,
                         verbose=verbose,
                         header='Arrays are not greater-ordered')


def assert_isreal(actual, err_msg='', verbose=True):
    """
    Raises an AssertionError if object is not real.

    The test is equivalent to ``isreal(actual)``.

    Parameters
    ----------
    actual : array_like
        Array obtained.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual is not real.

    See Also
    --------
    assert_allclose
    """

    import numpy as np
    np.testing.assert_equal(np.isreal(actual), True, err_msg, verbose)


def assert_array_not_equal(x, y, err_msg='', verbose=True):
    """
    Raises an AssertionError if two array_like objects are equal.

    Given two array_like objects, check that the shape is equal and all
    elements of these objects are not equal. An exception is raised at
    shape mismatch or conflicting values. In contrast to the standard usage
    in numpy, NaNs are compared like numbers, no assertion is raised if
    both objects have NaNs in the same positions (ToDo: Check 2 NaNs).

    The usual caution for verifying equality with floating point numbers is
    advised.

    Parameters
    ----------
    x : array_like
        The actual object to check.
    y : array_like
        The desired, expected object.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired objects are equal.

    See Also
    --------
    assert_array_equal

    """
    assert_array_compare(operator.__ne__, x, y, err_msg=err_msg,
                         verbose=verbose, header='Arrays are equal')


def assert_cosine_similarity(x, y, atol=1e-6):
    x_normalized = normalize_vector_to_unit_length(x)
    y_normalized = normalize_vector_to_unit_length(y)
    distance = 1 - np.abs(vector_H_vector(x_normalized, y_normalized)) ** 2
    assert_array_less(distance, atol)


def assert_hermitian(matrix, axes=(-2, -1)):
    np.testing.assert_allclose(matrix,
                               matrix.swapaxes(*axes[::-1]).conj())


def assert_positive_semidefinite(matrix):
    # https://en.wikipedia.org/wiki/Positive-definite_matrix

    # ToDo: make axes to a parameter
    axes = (-2, -1)

    # ToDo: Implement for non hermitian matrix
    assert_hermitian(matrix, axes)
    if axes == (-2, -1):
        eigenvalues, _ = np.linalg.eigh(matrix)
        assert_array_greater_equal(eigenvalues + 1e-6, 0)
    else:
        raise NotImplementedError()
