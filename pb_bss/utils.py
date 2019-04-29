import numpy as np
import scipy.linalg
import functools
import inspect
import warnings


# NOTE(kgriffs): We don't want our deprecations to be ignored by default,
# so create our own type.
class DeprecatedWarning(UserWarning):
    pass


def deprecated(instructions):
    """
    Original: https://gist.github.com/kgriffs/8202106

    Flags a method as deprecated.
    Args:
        instructions: A human-friendly string of instructions, such
            as: 'Please migrate to add_proxy() ASAP.'
    """
    def decorator(func):
        """This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = 'Call to deprecated function {} ({}). {}'.format(
                func.__qualname__,
                inspect.getfile(func),
                instructions)

            frame = inspect.currentframe().f_back

            warnings.warn_explicit(message,
                                   category=DeprecatedWarning,
                                   filename=inspect.getfile(frame.f_code),
                                   lineno=frame.f_lineno)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def _normalize(op):
    op = op.replace(',', '')
    op = op.replace(' ', '')
    op = ' '.join(c for c in op)
    op = op.replace(' * ', '*')
    op = op.replace('- >', '->')
    return op


def _only_reshape(array, source, target):
    source, target = source.split(), target.replace(' * ', '*').split()
    input_shape = {key: array.shape[index] for index, key in enumerate(source)}

    output_shape = []
    for t in target:
        product = 1
        if not t == '1':
            t = t.split('*')
            for t_ in t:
                product *= input_shape[t_]
        output_shape.append(product)

    return array.reshape(output_shape)


def reshape(array, operation):
    """ This is an experimental version of a generalized reshape.

    See test cases for examples.
    """
    operation = _normalize(operation)

    if '*' in operation.split('->')[0]:
        raise NotImplementedError(
            'Unflatten operation not supported by design. '
            'Actual values for dimensions are not available to this function.'
        )

    # Initial squeeze
    squeeze_operation = operation.split('->')[0].split()
    for axis, op in reversed(list(enumerate(squeeze_operation))):
        if op == '1':
            array = np.squeeze(array, axis=axis)

    # Transpose
    transposition_operation = operation.replace('1', ' ').replace('*', ' ')
    try:
        array = np.einsum(transposition_operation, array)
    except ValueError as e:
        msg = 'op: {}, shape: {}'.format(transposition_operation,
                                         np.shape(array))
        if len(e.args) == 1:
            e.args = (e.args[0] + '\n\n' + msg,)
        else:
            print(msg)
        raise

    # Final reshape
    source = transposition_operation.split('->')[-1]
    target = operation.split('->')[-1]

    return _only_reshape(array, source, target)


def get_pca(target_psd_matrix, use_scipy=False):
    """

    >>> M = np.array([[2, 0], [0, 1]])
    >>> get_pca(M, use_scipy=True)
    (array([1., 0.]), array(2.))
    >>> get_pca(M, use_scipy=False)
    (array([1., 0.]), array(2.))

    >>> M = np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]])
    >>> get_pca(M, use_scipy=True)
    (array([1., 0., 0.]), array(2.))
    >>> get_pca(M, use_scipy=False)
    (array([1., 0., 0.]), array(2.))


    """
    D = target_psd_matrix.shape[-1]

    # Save the shape of target_psd_matrix
    shape = target_psd_matrix.shape

    # Reduce independent dims to 1 independent dim
    target_psd_matrix = np.reshape(target_psd_matrix, (-1,) + shape[-2:])

    if use_scipy:
        beamforming_vector = []
        eigenvalues = []
        for f in range(target_psd_matrix.shape[0]):
            eigenvals, eigenvecs = scipy.linalg.eigh(
                target_psd_matrix[-1], eigvals=(D-1, D-1)
            )
            eigenval, = eigenvals
            eigenvec, = eigenvecs.T
            beamforming_vector.append(eigenvec)
            eigenvalues.append(eigenval)

        beamforming_vector = np.array(beamforming_vector)
        eigenvalues = np.array(eigenvalues)
    else:
        # Calculate eigenvals/vecs
        try:
            eigenvals, eigenvecs = np.linalg.eigh(target_psd_matrix)
        except np.linalg.LinAlgError:
            # ToDo: figure out when this happen and why eig may work.
            # It is likely that eig is more stable than eigh.
            eigenvals, eigenvecs = np.linalg.eig(target_psd_matrix)
            eigenvals = eigenvals.real

        # Select eigenvec for max eigenval. Eigenvals are sorted in ascending order.
        beamforming_vector = eigenvecs[..., -1]
        eigenvalues = eigenvals[..., -1]
        # Reconstruct original shape

    beamforming_vector = np.reshape(beamforming_vector, shape[:-1])
    eigenvalues = np.reshape(eigenvalues, shape[:-2])

    return beamforming_vector, eigenvalues


def get_power_spectral_density_matrix(
        observation, mask=None, sensor_dim=-2, source_dim=-2, time_dim=-1
):
    """
    Calculates the weighted power spectral density matrix.
    It's also called covariance matrix.
    With the dim parameters you can change the sort of the dims of the
    observation and mask.
    But not every combination is allowed.

    :param observation: Complex observations with shape (..., sensors, frames)
    :param mask: Masks with shape (bins, frames) or (..., sources, frames)
    :param sensor_dim: change sensor dimension index (Default: -2)
    :param source_dim: change source dimension index (Default: -2),
        source_dim = 0 means mask shape (sources, ..., frames)
    :param time_dim:  change time dimension index (Default: -1),
        this index must match for mask and observation
    :return: PSD matrix with shape (..., sensors, sensors)
        or (..., sources, sensors, sensors) or
        (sources, ..., sensors, sensors)
        if source_dim % observation.ndim < -2 respectively
        mask shape (sources, ..., frames)

    Examples
    --------
    >>> F, T, D, K = 51, 31, 6, 2
    >>> X = np.random.randn(F, D, T) + 1j * np.random.randn(F, D, T)
    >>> mask = np.random.randn(F, K, T)
    >>> mask = mask / np.sum(mask, axis=0, keepdims=True)
    >>> get_power_spectral_density_matrix(X, mask=mask).shape
    (51, 2, 6, 6)
    >>> mask = np.random.randn(F, T)
    >>> mask = mask / np.sum(mask, axis=0, keepdims=True)
    >>> get_power_spectral_density_matrix(X, mask=mask).shape
    (51, 6, 6)
    """

    # TODO: Can we use nt.utils.math_ops.covariance instead?

    # ensure negative dim indexes
    sensor_dim, source_dim, time_dim = (d % observation.ndim - observation.ndim
                                        for d in
                                        (sensor_dim, source_dim, time_dim))

    # ensure observation shape (..., sensors, frames)
    obs_transpose = [i for i in range(-observation.ndim, 0) if
                     i not in [sensor_dim, time_dim]] + [sensor_dim, time_dim]
    observation = observation.transpose(obs_transpose)

    if mask is None:
        psd = np.einsum('...dt,...et->...de', observation, observation.conj())

        # normalize
        psd /= observation.shape[-1]

    else:
        # Unfortunately, this function changes mask.
        mask = np.copy(mask)

        # normalize
        if mask.dtype == np.bool:
            mask = np.asfarray(mask)

        mask /= np.maximum(np.sum(mask, axis=time_dim, keepdims=True), 1e-10)

        if mask.ndim + 1 == observation.ndim:
            mask = np.expand_dims(mask, -2)
            psd = np.einsum('...dt,...et->...de', mask * observation,
                            observation.conj())
        else:
            # ensure shape (..., sources, frames)
            mask_transpose = [i for i in range(-observation.ndim, 0) if
                              i not in [source_dim, time_dim]] + [source_dim,
                                                                  time_dim]
            mask = mask.transpose(mask_transpose)

            psd = np.einsum('...kt,...dt,...et->...kde', mask, observation,
                            observation.conj())

            if source_dim < -2:
                # assume PSD shape (sources, ..., sensors, sensors) is interested
                psd = np.rollaxis(psd, -3, source_dim % observation.ndim)

    return psd


def get_stft_center_frequencies(size=1024, sample_rate=16000):
    """
    It is often necessary to know, which center frequency is
    represented by each frequency bin index.

    :param size: Scalar FFT-size.
    :param sample_rate: Scalar sample frequency in Hertz.
    :return: Array of all relevant center frequencies
    """
    frequency_index = np.arange(0, size/2 + 1)
    return frequency_index * sample_rate / size


def is_broadcast_compatible(*shapes):
    if len(shapes) < 2:
        return True
    else:
        for dim in zip(*[shape[::-1] for shape in shapes]):
            if len(set(dim).union({1})) <= 2:
                pass
            else:
                return False
        return True


def labels_to_one_hot(
        labels: np.ndarray, categories: int, axis: int = 0,
        keepdims=False, dtype=np.bool
):
    """ Translates an arbitrary ndarray with labels to one hot coded array.

    Args:
        labels: Array with any shape and integer labels.
        categories: Maximum integer label larger or equal to maximum of the
            labels ndarray.
        axis: Axis along which the one-hot vector will be aligned.
        keepdims:
            If keepdims is True, this function behaves similar to
            numpy.concatenate(). It will expand the provided axis.
            If keepdims is False, it will create a new axis along which the
            one-hot vector will be placed.
        dtype: Provides the dtype of the output one-hot mask.

    Returns:
        One-hot encoding with shape (..., categories, ...).

    >>> labels_to_one_hot([0, 1], categories=4)
    array([[ True, False],
           [False,  True],
           [False, False],
           [False, False]])
    >>> labels_to_one_hot([0, 1], categories=4, axis=-1)
    array([[ True, False, False, False],
           [False,  True, False, False]])
    >>> labels_to_one_hot([[0, 1], [0, 3]], categories=4, axis=-1)
    array([[[ True, False, False, False],
            [False,  True, False, False]],
    <BLANKLINE>
           [[ True, False, False, False],
            [False, False, False,  True]]])
    >>> labels_to_one_hot([[0, 1], [0, 3]], categories=4, axis=1)
    array([[[ True, False],
            [False,  True],
            [False, False],
            [False, False]],
    <BLANKLINE>
           [[ True, False],
            [False, False],
            [False, False],
            [False,  True]]])
    >>> labels_to_one_hot([[0, 1], [0, 3]], categories=4, axis=0)
    array([[[ True, False],
            [ True, False]],
    <BLANKLINE>
           [[False,  True],
            [False, False]],
    <BLANKLINE>
           [[False, False],
            [False, False]],
    <BLANKLINE>
           [[False, False],
            [False,  True]]])

    """
    labels = np.asarray(labels)

    if keepdims:
        assert labels.shape[axis] == 1
        result_ndim = labels.ndim
    else:
        result_ndim = labels.ndim + 1

    if axis < 0:
        axis += result_ndim

    shape = labels.shape
    zeros = np.zeros((categories, labels.size), dtype=dtype)
    zeros[labels.ravel(), range(labels.size)] = 1

    zeros = zeros.reshape((categories,) + shape)

    if keepdims:
        zeros = zeros[(slice(None),) * (axis + 1) + (0,)]

    zeros = np.moveaxis(zeros, 0, axis)

    return zeros


def unsqueeze(array, axis):
    """

    >>> unsqueeze(np.ones((2, 3)), (-3, -1)).shape
    (2, 1, 3, 1)

    >>> unsqueeze(13, (-2, -1)).shape
    (1, 1)

    Args:
        array:
        axis:

    Returns:

    """
    array = np.array(array)
    shape = list(np.shape(array))
    future_ndim = len(shape) + len(axis)

    try:
        np.empty((future_ndim,))[list(axis)]
    except IndexError as e:
        raise IndexError(np.shape(array), shape, axis) from e

    axis = [a % future_ndim for a in axis]
    for p in sorted(axis):
        shape.insert(p, 1)

    return np.reshape(array, shape)
