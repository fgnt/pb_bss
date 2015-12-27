from scipy.linalg import eig
import numpy as np
from scipy.linalg import eigh
from numpy.linalg import solve


def get_power_spectral_density_matrix_old(observation, mask=None):
    """
    Calculates the weighted power spectral density matrix.

    This does not yet work with more than one target mask.

    :param observation: Complex observations with shape (bins, sensors, frames)
    :param mask: Masks with shape (bins, frames) or (bins, 1, frames)
    :return: PSD matrix with shape (bins, sensors, sensors)
    """
    bins, sensors, frames = observation.shape

    if mask is None:
        mask = np.ones((bins, frames))
    if mask.ndim == 2:
        mask = mask[:, np.newaxis, :]

    mask /= np.maximum(np.sum(mask, axis=-1, keepdims=True), 1e-6)

    psd = np.einsum('...ft,...dt,...et->...de', mask, observation, observation.conj())
    # psd /= normalization[:, np.newaxis, np.newaxis]
    return psd


def get_power_spectral_density_matrix(observation, mask=None, sensor_dim=-2, source_dim=-2, time_dim=-1):
    """
    Calculates the weighted power spectral density matrix.
    It's also called covariance matrix.

    With the *_dim parameters you can change the sort of the dims of the observation and mask.
    But not every combination is allowed.

    :param observation: Complex observations with shape (..., sensors, frames)
    :param mask: Masks with shape (bins, frames) or (..., sources, frames)
    :param sensor_dim: change sensor dimension index (Default: -2)
    :param source_dim: change source dimension index (Default: -2)
    :param time_dim:  change time dimension index (Default: -1), this index must match for mask and observation
    :return: PSD matrix with shape (..., sensors, sensors) or (..., sources, sensors, sensors)

    Examples:
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

    if mask is None:
        if time_dim == -1 and sensor_dim == -2:
            psd = np.einsum('...dt,...et->...de', observation, observation.conj())
            psd /= observation.shape[-1]
        elif time_dim == -2 and sensor_dim == -1:
            psd = np.einsum('...td,...te->...de', observation, observation.conj())
            psd /= observation.shape[-2]
        elif time_dim == 0 and sensor_dim == 1:
            psd = np.einsum('td...,te...->de...', observation, observation.conj())
            psd /= observation.shape[0]
        else:
            print('time_dim: ', time_dim)
            print('sensor_dim: ', sensor_dim)
            print('observation.shape: ', observation.shape)
            raise NotImplementedError()
    else:
        mask /= np.maximum(np.sum(mask, axis=time_dim, keepdims=True), 1e-10)

        if mask.ndim + 1 == observation.ndim:
            mask = np.expand_dims(mask, sensor_dim)
            source_dim = None
        else:
            mask = np.rollaxis(mask, source_dim, sensor_dim)

        if time_dim == -1 and sensor_dim == -2:
            psd = np.einsum('...kt,...dt,...et->...kde', mask, observation, observation.conj())
            if source_dim is None:
                psd = np.squeeze(psd, axis=-3)
        elif time_dim == -2 and sensor_dim == -1:
            psd = np.einsum('...tk,...td,...te->...kde', mask, observation, observation.conj())
            if source_dim is None:
                psd = np.squeeze(psd, axis=-3)
        else:
            raise NotImplementedError()

    return psd


def get_pca_vector_old(target_psd_matrix):
    """
    Returns the beamforming vector of a PCA beamformer.
    :param target_psd_matrix: Target PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (bins, sensors)
    """
    bins, sensors, _ = target_psd_matrix.shape
    beamforming_vector = np.empty((bins, sensors), dtype=np.complex)
    for f in range(bins):
        eigenvals, eigenvecs = eigh(target_psd_matrix[f, :, :])
        beamforming_vector[f, :] = eigenvecs[:, np.argmax(eigenvals)]
    return beamforming_vector


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
    beamforming_vector = np.array([eigenvecs[i, :, vals[i]] for i in range(eigenvals.shape[0])])
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
    noise_psd_matrix = 0.5 * (noise_psd_matrix + np.conj(noise_psd_matrix.swapaxes(-1, -2)))

    numerator = solve(noise_psd_matrix, atf_vector)
    denominator = np.einsum('...d,...d->...', atf_vector.conj(), numerator)
    beamforming_vector = numerator / np.expand_dims(denominator, axis=-1)

    return beamforming_vector


def get_mvdr_vector_old(atf_vector, noise_psd_matrix):
    """
    Returns the MVDR beamforming vector.

    :param atf_vector: Acoustic transfer function vector
        with shape (bins, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (bins, sensors)
    """

    if atf_vector.ndim == 1:
        atf_vector = atf_vector[np.newaxis, :]
    if noise_psd_matrix.ndim == 2:
        noise_psd_matrix = noise_psd_matrix[np.newaxis, :, :]

    # Make sure matrix is hermitian
    noise_psd_matrix = 1 / 2 * (
        noise_psd_matrix + noise_psd_matrix.transpose(0, 2, 1).conj())

    bins, sensors = atf_vector.shape
    beamforming_vector = np.empty((bins, sensors), dtype=np.complex)
    for f in range(bins):
        numerator = solve(noise_psd_matrix[f, :, :], atf_vector[f, :])
        denominator = np.dot(atf_vector[f, :].conj(), numerator)
        beamforming_vector[f, :] = numerator / denominator
    return beamforming_vector


def get_gev_vector(target_psd_matrix, noise_psd_matrix):
    """
    Returns the GEV beamforming vector.
    :param target_psd_matrix: Target PSD matrix
        with shape (bins, sensors, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (bins, sensors)
    """
    bins, sensors, _ = target_psd_matrix.shape
    beamforming_vector = np.empty((bins, sensors), dtype=np.complex)
    for f in range(bins):
        try:
            eigenvals, eigenvecs = eigh(target_psd_matrix[f, :, :],
                                        noise_psd_matrix[f, :, :])
        except np.linalg.LinAlgError:
            eigenvals, eigenvecs = eig(target_psd_matrix[f, :, :],
                                       noise_psd_matrix[f, :, :])
        beamforming_vector[f, :] = eigenvecs[:, np.argmax(eigenvals)]
    return beamforming_vector


def get_lcmv_vector_old(atf_vectors, response_vector, noise_psd_matrix):
    """

    :param atf_vectors: Acoustic transfer function vectors for
        each source with shape (bins, targets, sensors)
    :param response_vector: Defines, which sources you are interested in.
        Set it to [1, 0, ..., 0], if you are interested in the first speaker.
        It has the shape (targets,)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (bins, sensors)
    """

    from scipy.linalg import solve as sci_solve

    if atf_vectors.ndim == 2:
        atf_vectors = atf_vectors[np.newaxis, :, :]
    if noise_psd_matrix.ndim == 2:
        noise_psd_matrix = noise_psd_matrix[np.newaxis, :, :]

    bins, targets, sensors = atf_vectors.shape
    beamforming_vector = np.empty((bins, sensors), dtype=np.complex)
    for f in range(bins):
        Phi_inverse_times_H = sci_solve(noise_psd_matrix[f, :, :], atf_vectors[f, :, :].transpose())
        H_times_Phi_inverse_times_H = np.dot(atf_vectors[f, :, :].conj(), Phi_inverse_times_H)
        beamforming_vector[f, :] = np.dot(Phi_inverse_times_H,
                                          solve(H_times_Phi_inverse_times_H, response_vector))

    return beamforming_vector


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

    Phi_inverse_times_H = solve(np.expand_dims(noise_psd_matrix, axis=0), atf_vectors)
    H_times_Phi_inverse_times_H = np.einsum('k...d,l...d->...kl', atf_vectors.conj(), Phi_inverse_times_H)
    temp = solve(H_times_Phi_inverse_times_H, np.expand_dims(response_vector, axis=0))
    beamforming_vector = np.einsum('k...d,...k->...d', Phi_inverse_times_H, temp)

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


def gev_wrapper_on_masks(mix, noise_mask=None, target_mask=None,
                         normalization=False):
    if noise_mask is None and target_mask is None:
        raise ValueError('At least one mask needs to be present.')

    mix = mix.T
    if noise_mask is not None:
        noise_mask = noise_mask.T
    if noise_mask is not None:
        target_mask = target_mask.T

    if target_mask is None:
        target_mask = np.clip(1 - noise_mask, 1e-6, 1)
    if noise_mask is None:
        noise_mask = np.clip(1 - target_mask, 1e-6, 1)

    bins, sensors, frames = mix.shape

    target_psd_matrix = get_power_spectral_density_matrix(mix, target_mask)
    noise_psd_matrix = get_power_spectral_density_matrix(mix, noise_mask)

    # Beamforming vector
    W_gev = get_gev_vector(target_psd_matrix, noise_psd_matrix)

    if normalization:
        W_gev = blind_analytic_normalization(W_gev, noise_psd_matrix)

    output = apply_beamforming_vector(W_gev, mix)

    return output.T
