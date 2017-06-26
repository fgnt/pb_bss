import numpy as np


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


def get_pca(target_psd_matrix):
    # Save the shape of target_psd_matrix
    shape = target_psd_matrix.shape

    # Reduce independent dims to 1 independent dim
    target_psd_matrix = np.reshape(target_psd_matrix, (-1,) + shape[-2:])

    # Calculate eigenvals/vecs
    eigenvals, eigenvecs = np.linalg.eigh(target_psd_matrix)
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
