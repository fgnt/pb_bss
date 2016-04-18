import warnings

import numpy as np
from numpy.linalg import solve
from numpy.linalg import eigh
from scipy.linalg import eig
from scipy.linalg import eigh

try:
    from .cythonized.get_gev_vector import _c_get_gev_vector
except ImportError:
    c_gev_available = False
    warnings.warn('Could not import cythonized get_gev_vector. Falling back to '
                  'python implementation. Maybe you need to rebuild/reinstall '
                  'the toolbox?')
else:
    c_gev_available = True


def get_power_spectral_density_matrix(observation, mask=None, sensor_dim=-2,
                                      source_dim=-2, time_dim=-1):
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


def get_pca_vector(target_psd_matrix):
    """
    Returns the beamforming vector of a PCA beamformer.

    :param target_psd_matrix: Target PSD matrix
        with shape (..., sensors, sensors)
    :return: Set of beamforming vectors with shape (..., sensors)
    """
    # Save the shape of target_psd_matrix
    shape = target_psd_matrix.shape

    # Reduce independent dims to 1 independent dim
    target_psd_matrix = np.reshape(target_psd_matrix, (-1,) + shape[-2:])

    # Calculate eigenvals/vecs
    eigenvals, eigenvecs = np.linalg.eigh(target_psd_matrix)
    # Find max eigenvals
    vals = np.argmax(eigenvals, axis=-1)
    # Select eigenvec for max eigenval
    beamforming_vector = np.array(
        [eigenvecs[i, :, vals[i]] for i in range(eigenvals.shape[0])]
    )
    # Reconstruct original shape
    beamforming_vector = np.reshape(beamforming_vector, shape[:-1])

    return beamforming_vector


# TODO: Possible test case: Assert W^H * H = 1.
# TODO: Make function more stable for badly conditioned noise matrices.
# Write tests for these cases.
def get_mvdr_vector(atf_vector, noise_psd_matrix):
    """
    Returns the MVDR beamforming vector.

    :param atf_vector: Acoustic transfer function vector
        with shape (..., bins, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (..., bins, sensors)
    """

    while atf_vector.ndim > noise_psd_matrix.ndim - 1:
        noise_psd_matrix = np.expand_dims(noise_psd_matrix, axis=0)

    # Make sure matrix is hermitian
    noise_psd_matrix = 0.5 * (
        noise_psd_matrix + np.conj(noise_psd_matrix.swapaxes(-1, -2))
    )

    numerator = solve(noise_psd_matrix, atf_vector)
    denominator = np.einsum('...d,...d->...', atf_vector.conj(), numerator)
    beamforming_vector = numerator / np.expand_dims(denominator, axis=-1)

    return beamforming_vector


def get_gev_vector(target_psd_matrix, noise_psd_matrix, force_cython=False):
    """
    Returns the GEV beamforming vector.

    :param target_psd_matrix: Target PSD matrix
        with shape (bins, sensors, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (bins, sensors)
    """
    if c_gev_available:
        try:
            return _c_get_gev_vector(
                np.asfortranarray(target_psd_matrix.astype(np.complex128).T),
                np.asfortranarray(noise_psd_matrix.astype(np.complex128).T))
        except ValueError as e:
            if not force_cython:
                pass
            else:
                raise e
    return _get_gev_vector(target_psd_matrix, noise_psd_matrix)


def _get_gev_vector(target_psd_matrix, noise_psd_matrix):
    assert target_psd_matrix.shape == noise_psd_matrix.shape
    assert target_psd_matrix.shape[-2] == target_psd_matrix.shape[-1]

    sensors = target_psd_matrix.shape[-1]

    original_shape = target_psd_matrix.shape
    target_psd_matrix = target_psd_matrix.reshape((-1, sensors, sensors))
    noise_psd_matrix = noise_psd_matrix.reshape((-1, sensors, sensors))

    bins = target_psd_matrix.shape[0]
    beamforming_vector = np.empty((bins, sensors), dtype=np.complex)

    for f in range(bins):
        try:
            eigenvals, eigenvecs = eigh(
                target_psd_matrix[f, :, :], noise_psd_matrix[f, :, :]
            )
        except np.linalg.LinAlgError:
            eigenvals, eigenvecs = eig(
                target_psd_matrix[f, :, :], noise_psd_matrix[f, :, :]
            )
        beamforming_vector[f, :] = eigenvecs[:, np.argmax(eigenvals)]

    return beamforming_vector.reshape(original_shape[:-1])


def get_lcmv_vector(atf_vectors, response_vector, noise_psd_matrix):
    """

    :param atf_vectors: Acoustic transfer function vectors for
        each source with shape (targets, bins, sensors)
    :param response_vector: Defines, which sources you are interested in.
        Set it to [1, 0, ..., 0], if you are interested in the first speaker.
        It has the shape (targets,)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (bins, sensors)
    """

    Phi_inverse_times_H = solve(
        np.expand_dims(noise_psd_matrix, axis=0),
        atf_vectors
    )
    H_times_Phi_inverse_times_H = np.einsum(
        'k...d,l...d->...kl',
        atf_vectors.conj(),
        Phi_inverse_times_H
    )
    temp = solve(
        H_times_Phi_inverse_times_H,
        np.expand_dims(response_vector, axis=0)
    )
    beamforming_vector = np.einsum(
        'k...d,...k->...d',
        Phi_inverse_times_H,
        temp
    )

    return beamforming_vector


def blind_analytic_normalization(vector, noise_psd_matrix):
    bins, sensors = vector.shape
    normalization = np.zeros(bins)
    for f in range(bins):
        normalization[f] = np.abs(np.sqrt(np.dot(
            np.dot(np.dot(vector[f, :].T.conj(), noise_psd_matrix[f]),
                   noise_psd_matrix[f]), vector[f, :])))
        normalization[f] /= np.abs(np.dot(
            np.dot(vector[f, :].T.conj(), noise_psd_matrix[f]), vector[f, :]))

    return vector * normalization[:, np.newaxis]


def apply_beamforming_vector(vector, mix):
    """Applies a beamforming vector such that the sensor dimension disappears.

    :param vector: Beamforming vector with dimensions ..., sensors
    :param mix: Observed signal with dimensions ..., sensors, time-frames
    :return: A beamformed signal with dimensions ..., time-frames
    """
    return np.einsum('...a,...at->...t', vector.conj(), mix)


def apply_online_beamforming_vector(vector, mix):
    """Applies a beamforming vector such that the sensor dimension disappears.

    :param vector: Beamforming vector with dimensions ..., sensors
    :param mix: Observed signal with dimensions ..., sensors, time-frames
    :return: A beamformed signal with dimensions ..., time-frames
    """
    vector = vector.transpose(1, 2, 0)
    return np.einsum('...at,...at->...t', vector.conj(), mix)


def gev_wrapper_on_masks(mix, noise_mask=None, target_mask=None,
                         normalization=False):
    if noise_mask is None and target_mask is None:
        raise ValueError('At least one mask needs to be present.')

    mix = mix.T
    if noise_mask is not None:
        noise_mask = noise_mask.T
    if target_mask is not None:
        target_mask = target_mask.T

    if target_mask is None:
        target_mask = np.clip(1 - noise_mask, 1e-6, 1)
    if noise_mask is None:
        noise_mask = np.clip(1 - target_mask, 1e-6, 1)

    target_psd_matrix = get_power_spectral_density_matrix(mix, target_mask)
    noise_psd_matrix = get_power_spectral_density_matrix(mix, noise_mask)

    W_gev = get_gev_vector(target_psd_matrix, noise_psd_matrix)

    if normalization:
        W_gev = blind_analytic_normalization(W_gev, noise_psd_matrix)

    output = apply_beamforming_vector(W_gev, mix)

    return output.T


def pca_wrapper_on_masks(mix, noise_mask=None, target_mask=None):
    if noise_mask is None and target_mask is None:
        raise ValueError('At least one mask needs to be present.')

    mix = mix.T
    if noise_mask is not None:
        noise_mask = noise_mask.T
    if target_mask is not None:
        target_mask = target_mask.T

    if target_mask is None:
        target_mask = np.clip(1 - noise_mask, 1e-6, 1)

    target_psd_matrix = get_power_spectral_density_matrix(mix, target_mask)

    W_pca = get_pca_vector(target_psd_matrix)

    output = apply_beamforming_vector(W_pca, mix)

    return output.T


def pca_mvdr_wrapper_on_masks(mix, noise_mask=None, target_mask=None,
                              regularization=None):
    if noise_mask is None and target_mask is None:
        raise ValueError('At least one mask needs to be present.')

    mix = mix.T
    if noise_mask is not None:
        noise_mask = noise_mask.T
    if target_mask is not None:
        target_mask = target_mask.T

    if target_mask is None:
        target_mask = np.clip(1 - noise_mask, 1e-6, 1)
    if noise_mask is None:
        noise_mask = np.clip(1 - target_mask, 1e-6, 1)

    target_psd_matrix = get_power_spectral_density_matrix(mix, target_mask)
    noise_psd_matrix = get_power_spectral_density_matrix(mix, noise_mask)

    if regularization is not None:
        noise_psd_matrix += np.tile(
            regularization * np.eye(noise_psd_matrix.shape[1]),
            (noise_psd_matrix.shape[0], 1, 1)
        )

    W_pca = get_pca_vector(target_psd_matrix)
    W_mvdr = get_mvdr_vector(W_pca, noise_psd_matrix)

    output = apply_beamforming_vector(W_mvdr, mix)

    return output.T
