""" Beamformer module.

The shape convention is to place time at the end to speed up computation and
move independent dimensions to the front.

That results i.e. in the following possible shapes:
    X: Shape (F, D, T).
    mask: Shape (F, K, T).
    PSD: Shape (F, D, D).

The functions themselves are written more generic, though.
"""

import warnings
from scipy.linalg import sqrtm
import numpy as np
from numpy.linalg import solve
from scipy.linalg import eig
from scipy.linalg import eigh
from nt.math.correlation import covariance  # as shortcut!

try:
    from .cythonized.get_gev_vector import _c_get_gev_vector
except ImportError:
    c_gev_available = False
    warnings.warn('Could not import cythonized get_gev_vector. Falling back to '
                  'python implementation. Maybe you need to rebuild/reinstall '
                  'the toolbox?')
else:
    c_gev_available = True

try:
    from .cythonized.c_eig import _cythonized_eig
except ImportError:
    c_eig_available = False
    warnings.warn('Could not import cythonized eig. Falling back to '
                  'python implementation. Maybe you need to rebuild/reinstall '
                  'the toolbox?')
else:
    c_eig_available = True


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


def get_pca_vector(target_psd_matrix):
    """
    Returns the beamforming vector of a PCA beamformer.

    :param target_psd_matrix: Target PSD matrix
        with shape (..., sensors, sensors)
    :return: Set of beamforming vectors with shape (..., sensors)
    """
    return get_pca(target_psd_matrix)[0]


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
    try:
        numerator = solve(noise_psd_matrix, atf_vector)
    except np.linalg.LinAlgError:
        bins = noise_psd_matrix.shape[0]
        numerator = np.empty_like(atf_vector)
        for f in range(bins):
            numerator[f], *_ = np.linalg.lstsq(noise_psd_matrix[f],
                                               atf_vector[..., f, :])
    denominator = np.einsum('...d,...d->...', atf_vector.conj(), numerator)
    beamforming_vector = numerator / np.expand_dims(denominator, axis=-1)

    return beamforming_vector


def get_mvdr_vector_merl(target_psd_matrix, noise_psd_matrix):
    """
    Returns the MVDR beamforming vector.

    This implementation is based on a variant described in 
    https://www.merl.com/publications/docs/TR2016-072.pdf
    It selects a reference channel that maximizes the post-SNR.

    :param target_psd_matrix: Target PSD matrix
        with shape (..., bins, sensors, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (..., bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (..., bins, sensors)
    """
    G = np.linalg.solve(noise_psd_matrix, target_psd_matrix)
    lambda_ = np.trace(G, axis1=-2, axis2=-1)
    h = G / lambda_[..., None, None]

    nom = np.sum(
        np.einsum('...fac,fab,...fbc->c', h.conj(), target_psd_matrix, h)
    )
    denom = np.sum(
        np.einsum('...fac,fab,...fbc->c', h.conj(), noise_psd_matrix, h)
    )
    h_idx = np.argmax(nom/denom)

    return h[..., h_idx]


def get_gev_vector(target_psd_matrix, noise_psd_matrix, force_cython=False,
                   use_eig=False):
    """
    Returns the GEV beamforming vector.

    :param target_psd_matrix: Target PSD matrix
        with shape (..., sensors, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (..., sensors, sensors)
    :return: Set of beamforming vectors with shape (..., sensors)
    """
    if c_gev_available and not use_eig:
        try:
            if target_psd_matrix.ndim == 3:
                return _c_get_gev_vector(
                    np.asfortranarray(target_psd_matrix.astype(np.complex128).T),
                    np.asfortranarray(noise_psd_matrix.astype(np.complex128).T))
            else:
                D = target_psd_matrix.shape[-1]
                assert D == target_psd_matrix.shape[-2]
                assert target_psd_matrix.shape == noise_psd_matrix.shape
                dst_shape = target_psd_matrix.shape[:-1]
                target_psd_matrix = target_psd_matrix.reshape(-1, D, D)
                noise_psd_matrix = noise_psd_matrix.reshape(-1, D, D)
                ret = _c_get_gev_vector(
                    np.asfortranarray(target_psd_matrix.astype(np.complex128).T),
                    np.asfortranarray(noise_psd_matrix.astype(np.complex128).T))
                return ret.reshape(*dst_shape)
        except ValueError as e:
            if not force_cython:
                pass
            else:
                raise e
    if c_eig_available and use_eig:
        try:
            eigenvals_c, eigenvecs_c = _cythonized_eig(
                target_psd_matrix, noise_psd_matrix)
            return eigenvecs_c[
                   range(target_psd_matrix.shape[0]), :,
                   np.argmax(eigenvals_c, axis=1)]
        except ValueError as e:
            if not force_cython:
                pass
            else:
                raise e
    return _get_gev_vector(target_psd_matrix, noise_psd_matrix, use_eig)


def _get_gev_vector(target_psd_matrix, noise_psd_matrix, use_eig=False):
    assert target_psd_matrix.shape == noise_psd_matrix.shape
    assert target_psd_matrix.shape[-2] == target_psd_matrix.shape[-1]

    sensors = target_psd_matrix.shape[-1]

    original_shape = target_psd_matrix.shape
    target_psd_matrix = target_psd_matrix.reshape((-1, sensors, sensors))
    noise_psd_matrix = noise_psd_matrix.reshape((-1, sensors, sensors))

    bins = target_psd_matrix.shape[0]
    beamforming_vector = np.empty((bins, sensors), dtype=np.complex128)

    solver = eig if use_eig else eigh

    for f in range(bins):
        try:
            eigenvals, eigenvecs = solver(
                target_psd_matrix[f, :, :], noise_psd_matrix[f, :, :]
            )
        except ValueError:
            raise ValueError('Error for frequency {}\n'
                             'phi_xx: {}\n'
                             'phi_nn: {}'.format(
                f, target_psd_matrix[f], noise_psd_matrix[f]))
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError('Error for frequency {}\n'
                             'phi_xx: {}\n'
                             'phi_nn: {}'.format(
                f, target_psd_matrix[f], noise_psd_matrix[f]))
        beamforming_vector[f, :] = eigenvecs[:, np.argmax(eigenvals)]

    return beamforming_vector.reshape(original_shape[:-1])


def get_lcmv_vector(atf_vectors, response_vector, noise_psd_matrix):
    """

    :param atf_vectors: Acoustic transfer function vectors for
        each source with shape (targets k, bins f, sensors d)
    :param response_vector: Defines, which sources you are interested in.
        Set it to [1, 0, ..., 0], if you are interested in the first speaker.
        It has the shape (targets,)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins f, sensors d, sensors D)
    :return: Set of beamforming vectors with shape (bins f, sensors d)
    """
    response_vector = np.asarray(response_vector)
    # TODO: If it is a list, a list of response_vectors is returned.

    Phi_inverse_times_H = solve(
        noise_psd_matrix[None, ...],  # 1, f, d, D
        atf_vectors  # k, f, d
    )  # k, f, d

    H_times_Phi_inverse_times_H = np.einsum(
        'k...d,K...d->...kK',
        atf_vectors.conj(),
        Phi_inverse_times_H
    )  # f, k, K

    temp = solve(
        H_times_Phi_inverse_times_H,
        response_vector[None, ...],  # 1, K
    )  # f, k
    beamforming_vector = np.einsum(
        'k...d,...k->...d',
        Phi_inverse_times_H,
        temp
    )

    return beamforming_vector


def blind_analytic_normalization(vector, noise_psd_matrix,
                                 target_psd_matrix=None):
    """Reduces distortions in beamformed ouptput.
    Args:
        vector: Beamforming vector with shape (..., sensors)
        noise_psd_matrix: With shape (..., sensors, sensors)
    """
    nominator = np.einsum(
        '...a,...ab,...bc,...c->...',
        vector.conj(), noise_psd_matrix, noise_psd_matrix, vector
    )
    if target_psd_matrix is not None:
        atf = get_pca_vector(target_psd_matrix)
        nominator /= atf
    nominator = np.sqrt(nominator)

    denominator = np.einsum(
        '...a,...ab,...b->...', vector.conj(), noise_psd_matrix, vector
    )
    denominator = np.sqrt(denominator * denominator.conj())
    denominator[denominator==0] = 1e-8
    normalization = np.abs(nominator / (denominator))
    return vector * normalization[..., np.newaxis]


def distortionless_normalization(vector, atf_vector, noise_psd_matrix):
    nominator = np.einsum(
        'fab,fb,fc->fac', noise_psd_matrix, vector, vector.conj()
    )
    denominator = np.einsum(
        'fa,fab,fb->f', vector.conj(), noise_psd_matrix, vector
    )
    projection_matrix = nominator / denominator[..., None, None]
    return np.einsum('fab,fb->fa', projection_matrix, atf_vector)


def mvdr_snr_postfilter(vector, target_psd_matrix, noise_psd_matrix):
    nominator = np.einsum(
        'fa,fab,fb->f', vector.conj(), target_psd_matrix, vector
    )
    denominator = np.einsum(
        'fa,fab,fb->f', vector.conj(), noise_psd_matrix, vector
    )
    return (nominator / denominator)[:, None]


def zero_degree_normalization(vector, reference_channel):
    return vector * np.exp(-1j * np.tile(np.angle(vector[:, reference_channel]), (vector.shape[-1],1))).transpose()


def phase_correction(vector):
    """Phase correction to reduce distortions due to phase inconsistencies.

    We need a copy first, because not all elements are touched during the
    multiplication. Otherwise, the vector would be modified in place.

    TODO: Write test cases.
    TODO: Only use non-loopy version when test case is written.

    Args:
        vector: Beamforming vector with shape (..., bins, sensors).
    Returns: Phase corrected beamforming vectors. Lengths remain.

    >>> w = np.array([[1, 1], [-1, -1]], dtype=np.complex128)
    >>> np.around(phase_correction(w), decimals=14)
    array([[ 1.+0.j,  1.+0.j],
           [ 1.-0.j,  1.-0.j]])
    >>> np.around(phase_correction([w]), decimals=14)[0]
    array([[ 1.+0.j,  1.+0.j],
           [ 1.-0.j,  1.-0.j]])
    >>> w  # ensure that w is not modified
    array([[ 1.+0.j,  1.+0.j],
           [-1.+0.j, -1.+0.j]])
    """

    # w = W.copy()
    # F, D = w.shape
    # for f in range(1, F):
    #     w[f, :] *= np.exp(-1j*np.angle(
    #         np.sum(w[f, :] * w[f-1, :].conj(), axis=-1, keepdims=True)))
    # return w

    vector = np.array(vector, copy=True)
    vector[..., 1:, :] *= np.cumprod(
        np.exp(
            1j * np.angle(
                np.sum(
                    vector[..., 1:, :].conj() * vector[..., :-1, :],
                    axis=-1, keepdims=True
                )
            )
        ), axis=0
    )
    return vector


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

def get_mvdr_vector_souden(target_psd_matrix, noise_psd_matrix, eps=1e-5):
    """
    Returns the MVDR beamforming vector described in [Souden10].

    :param target_psd_matrix: Target PSD matrix
        with shape (bins, sensors, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (..., bins, sensors)
    """

    # Make sure matrix is hermitian
    noise_psd_matrix = 0.5 * (
        noise_psd_matrix + np.conj(noise_psd_matrix.swapaxes(-1, -2))
    )
    assert target_psd_matrix.shape == noise_psd_matrix.shape
    assert target_psd_matrix.shape[-2] == target_psd_matrix.shape[-1]
    sensors = target_psd_matrix.shape[-1]

    target_psd_matrix = target_psd_matrix.reshape((-1, sensors, sensors))
    noise_psd_matrix = noise_psd_matrix.reshape((-1, sensors, sensors))

    bins = target_psd_matrix.shape[0]
    numerator = np.empty((bins, sensors, sensors), dtype=np.complex128)
    for f in range(bins):
        try:
            numerator[f, :, :] = np.linalg.solve(
                noise_psd_matrix[f, :, :], target_psd_matrix[f, :, :])

        except ValueError:
            raise ValueError('Error for frequency {}\n'
                             'phi_xx: {}\n'
                             'phi_nn: {}'.format(
                f, target_psd_matrix[f], noise_psd_matrix[f]))
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError('Error for frequency {}\n'
                                        'phi_xx: {}\n'
                                        'phi_nn: {}'.format(
                f, target_psd_matrix[f], noise_psd_matrix[f]))
    denominator = np.trace(numerator, axis1=1, axis2=2)
    beamforming_vector = numerator[:, :, 0] / np.expand_dims(denominator + eps, axis=-1)

    return beamforming_vector


class PowerMethodEigenvalueTracking():
    def __init__(self, r=1, channels=6, num_bins=513, alpha=0.99, beta=0.99, eps=1e-10):
        self.forget = [alpha, beta]
        self.eps = eps
        self.r = r
        self.A_tilde = np.array([0.2*np.eye(channels, channels, dtype=np.complex128) for idx in range(num_bins)])
        self.B_tilde = np.array([0.2*np.eye(channels, channels, dtype=np.complex128) for idx in range(num_bins)])
        self.W_hat = np.ones((num_bins, channels, r), dtype=np.complex128)
        self.sum_n = np.zeros((num_bins,1,1))
        self.sum_s = np.zeros((num_bins,1,1))

    def __call__(self, Y, speech_mask, noise_mask):
        assert Y.shape[-2] > Y.shape[-1],\
            'The frequency dim is smaller than the channel dim,' \
            ' something must have gone wrong'

        if Y.ndim == 2:
            F, D = Y.shape
        elif Y.ndim == 3:
            T, F, D = Y.shape
        else: raise ValueError('Y.ndim = {} ist to large,'
                             ' onlly 2 or 3 dimensions are allowed'.format(Y.ndim))
        A_tilde = self.A_tilde
        B_tilde = self.B_tilde
        if speech_mask.ndim + 1 == Y.ndim:
            speech_mask = np.expand_dims(speech_mask, -1)
        if noise_mask.ndim + 1 == Y.ndim:
            noise_mask = np.expand_dims(noise_mask, -1)
        A_tilde = self.forget[0] * A_tilde + np.einsum('...d,...e->...de',
                                                                 speech_mask * Y, Y.conj())
        B_tilde = self.forget[1] * B_tilde + np.einsum('...d,...e->...de',
                                                                 noise_mask * Y, Y.conj())

        self.sum_s += np.expand_dims(speech_mask, axis=-1)
        self.sum_n += np.expand_dims(noise_mask, axis=-1)
        sum_s = np.maximum(self.sum_s, self.eps)
        A = A_tilde/sum_s
        B = B_tilde

        K = np.array([np.linalg.inv(sqrtm(b+np.eye(D,D)*self.eps)) for b in B])*self.sum_n
        C = np.einsum('...ad,...da,...ad->...ad', K, A, K.conj())
        W_tilde = np.einsum('...ad,...dr->...ar', C, self.W_hat)
        for f in range(F):
            Q, R = np.linalg.qr(W_tilde[f, :, :], 'complete')
            self.W_hat[f, :, :] = Q[:, :self.r]
        self.A_tilde = A_tilde
        self.B_tilde = B_tilde
        W =  np.einsum('...da,...dr->...ar', K.conj(), self.W_hat)[:, :, :self.r]
        if np.sum(self.A_tilde)<100:
            W = np.ones(W.shape)
            print(self.A_tilde)
        B = B / np.maximum(self.sum_n, self.eps)
        return W, B


class GeneralizedEigenvectorTracking():
    def __init__(self, r=1, channels=6, num_bins=513, alpha=0.99, beta=0.99, eps=1e-10):
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.r = r
        self.K = np.array(
            [.01 * np.eye(channels, channels, dtype=np.complex128) for idx in
             range(num_bins)])
        self.C = np.array(
            [0.2 * np.eye(channels, channels, dtype=np.complex128) for idx in
             range(num_bins)])
        self.W_hat = np.ones((num_bins, channels, r), dtype=np.complex128)
        self.B_tilde = np.array(
            [0.2 * np.eye(channels, channels, dtype=np.complex128) for idx in
             range(num_bins)])
        self.A_tilde = np.array(
            [0.2 * np.eye(channels, channels, dtype=np.complex128) for idx in
             range(num_bins)])
        self.sum_n = np.zeros((num_bins, 1)) + eps
        self.sum_s = np.zeros((num_bins, 1)) + eps

    def __call__(self, Y, speech_mask, noise_mask):
        assert Y.ndim == 2, 'Only frequency and channel dim allowed'
        assert Y.shape[0] > Y.shape[
            1], 'The frequency dim is smaller than the channel dim, something must have gone wrong'
        F, D = Y.shape
        if noise_mask.ndim + 1 == Y.ndim:
            noise_mask = np.expand_dims(np.sqrt(noise_mask), -1)
        if speech_mask.ndim + 1 == Y.ndim:
            speech_mask = np.expand_dims(np.sqrt(speech_mask), -1)
        self.sum_s += speech_mask**2
        self.sum_n += noise_mask**2
        # step 1
        x_tilde = np.einsum('...ab, ...b->...a', self.K, np.sqrt(speech_mask) * Y)
        n_tilde = 1 / np.sqrt(self.beta) * np.einsum('...ab, ...b->...a', self.K,
                                                     np.sqrt(noise_mask) * Y)
        n_hat = 1 / np.sqrt(self.beta) * np.einsum('...ab, ...b->...a', self.K.conj(),
                                                   n_tilde)
        z = np.einsum('...ab, ...b->...a', self.C, n_tilde)
        # step 2
        n_tilde_norm = np.einsum('...a,...a->...', n_tilde.conj(),
                                 n_tilde)  # /self.sum_n.squeeze()
        gamma = np.expand_dims(
            1 / (n_tilde_norm + self.eps) * (1 / np.sqrt(1 + n_tilde_norm) - 1), axis=-1)
        e = np.expand_dims(np.einsum('...a,...a->...', x_tilde.conj(), n_tilde), axis=-1)
        delta = np.expand_dims(np.abs(gamma) ** 2 * (self.alpha * np.expand_dims(
            np.einsum('...a,...a->...', n_tilde.conj(), z), axis=-1
        ) + np.abs(e) ** 2), axis=-1)
        # step 3
        h = self.alpha * z + e * x_tilde
        n_h = np.expand_dims(gamma, axis=-1) * np.einsum('...a,...b->...ab', n_tilde,
                                                         h.conj())
        # step 4
        A_tilde = np.einsum('...a,...b->...ab', x_tilde,
                            x_tilde.conj())# /np.maximum(
            # np.expand_dims(self.sum_s, axis=-1), self.eps)
        B_tilde = np.einsum('...a,...b->...ab', n_tilde,
                            n_tilde.conj())  # /np.expand_dims(self.sum_n, axis=-1)
        C = 1 / self.beta * (self.alpha * self.C + A_tilde
                             + delta * B_tilde + n_h
                             + n_h.conj().transpose(0, 2, 1))*np.expand_dims(self.sum_n, axis=1)
        # step 5
        W_tilde = np.einsum('...ab,...br->...ar', C, self.W_hat)
        # step 6
        for f in range(F):
            Q, R = np.linalg.qr(W_tilde[f, :, :], 'complete')
            self.W_hat[f, :, :] = Q[:, :self.r]
        # step 7
        self.K = 1 / np.sqrt(self.beta) * self.K + np.expand_dims(
            gamma, axis=-1) * np.einsum('...a,...b->...ab', n_tilde,
                                        n_hat.conj()) / np.expand_dims(self.sum_n,
                                                                       axis=-1)
        # step 8
        self.B_tilde = self.beta * self.B_tilde + np.einsum('...a,...b->...ab',
                                                            noise_mask * Y, Y.conj())
        sum_n = np.maximum(np.expand_dims(self.sum_n, axis=-1), self.eps)
        self.B = self.B_tilde / sum_n

        W = np.einsum('...ba,...br->...ar', self.K.conj(), self.W_hat)

        return W, self.B



class RLSEigenvectorTracking():
    def __init__(self, channels=6, num_bins=513, alpha=0.99, beta=0.99, Lx=50, Ln=200,
                 eps=1e-10, sliding_window=True):
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        assert Lx <= Ln, 'The decaying speech window should be smaller or equal compared to the noise window'
        self.Lx = Lx
        self.Ln = Ln
        self.Q = np.array(
            [.01 * np.eye(channels, channels, dtype=np.complex128) for idx in
             range(num_bins)])
        self.W = np.ones((num_bins, channels), dtype=np.complex128)
        self.B = np.array(
            [.01 * np.eye(channels, channels, dtype=np.complex128) for idx in
             range(num_bins)])
        self.X = np.zeros((Lx, num_bins, channels), dtype=np.complex128)
        self.N = np.zeros((Ln, num_bins, channels), dtype=np.complex128)
        self.Z = np.zeros((Lx, num_bins), dtype=np.complex128)
        self.P_z =np.zeros((1,num_bins), dtype=np.complex128)
        self.r = np.zeros((1, num_bins, channels), dtype=np.complex128)
        self.num_bins = num_bins
        self.idx = 0
        self.sum_s = np.zeros((1, num_bins, 1, 1), dtype=np.complex128)
        self.sum_n = np.zeros((1, num_bins, 1, 1), dtype=np.complex128)
        self.sliding_window = sliding_window

    def __call__(self, Y, speech_mask, noise_mask):
        assert Y.ndim == 2, 'Only frequency and channel dim allowed'
        assert Y.shape[0] > Y.shape[
            1], 'The frequency dim is smaller than the channel dim, something must have gone wrong'
        F, D = Y.shape
        if noise_mask.ndim + 1 == Y.ndim:
            noise_mask = np.expand_dims(noise_mask, -1)
        if speech_mask.ndim + 1 == Y.ndim:
            speech_mask = np.expand_dims(speech_mask, -1)
        if self.idx == self.Ln:
            self.idx = self.idx
        elif self.idx >= self.Lx:
            self.idx += 1
        else:
            self.idx += 1

        if self.sliding_window:
            self.sum_s = np.concatenate((speech_mask[None,:,:,None], self.sum_s), axis=0)
            self.sum_n = np.concatenate((noise_mask[None,:,:,None], self.sum_n), axis=0)
            # step 1
            Z = np.einsum('...a,...a->...', self.W.conj(), Y)
            # Z = speech_mask * Y
            # step 2
            self.P_z = self.alpha * self.P_z + Z * Z.conj() - self.alpha ** self.Lx * np.abs(
                self.Z[-1]) ** 2
            self.P_z[self.P_z==0] = self.eps
            P_z_inv = np.reciprocal(self.P_z)
            self.r = self.alpha * self.r + Y * np.expand_dims(
                Z,axis=-1).conj() - self.alpha ** self.Lx * self.X[-1] * np.expand_dims(
                self.Z[-1], axis=-1).conj()
            sum_s = np.sum(self.sum_s, axis=0)
            sum_s[sum_s==0] = 1e-8
            r = self.r #/ (sum_s)
            # step 3
            a = np.tile([[self.beta, 0], [0, self.beta ** (1 - self.Ln)]],
                        [self.num_bins, 1, 1])
            N_N = np.stack([noise_mask * Y, self.N[-1]], axis=-1)
            b = np.linalg.inv(
                a + np.einsum('...bd, ...bc, ...ce->...de', N_N.conj(), self.Q, N_N))
            c = np.einsum('...ab,...bc,...cd->...ad', self.Q, N_N, b)
            d = np.einsum('...ab,...cb,...cd->...ad', c, N_N.conj(), self.Q)
            self.Q = (self.Q - d) / self.beta
            Q = self.Q * np.sum(self.sum_n, axis=0)
            # step 4
            self.W = np.einsum('...ab,...b->...a', Q, r)[0] * np.expand_dims(
                P_z_inv, axis=-1)
            self.W = self.W[0]
        else:
            self.sum_s =+ speech_mask[None,:,:,None]
            self.sum_n =+ noise_mask[None,:,:,None]
            # step 1
            Z = np.einsum('...a,...a->...', self.W.conj(), Y)
            # Z =  Y
            # step 2
            self.P_z = self.alpha * self.P_z + Z * Z.conj()
            self.P_z[self.P_z==0] = self.eps
            P_z_inv = np.reciprocal(self.P_z)
            self.r = self.alpha * self.r + Y * np.expand_dims(
                Z,axis=-1).conj()
            # sum_s = np.sum(self.sum_s, axis=0)
            r = self.r #/ (sum_s + self.eps)
            # step 3
            a = np.tile([self.beta],
                        [self.num_bins, 1, 1])
            N = (np.sqrt(noise_mask) * Y)[:,:,None]
            b = np.linalg.inv(
                a + np.einsum('...bd, ...bc, ...ce->...de', N.conj(), self.Q, N))
            c = np.einsum('...ab,...bc,...cd->...ad', self.Q, N, b)
            d = np.einsum('...ab,...cb,...cd->...ad', c, N.conj(), self.Q)
            self.Q = (self.Q - d) / self.beta
            Q = self.Q * np.sum(self.sum_n[0], axis=0)
            # step 4
            self.W = np.einsum('...ab,...b->...a', Q, r)[0] * np.expand_dims(
                P_z_inv, axis=-1)
            self.W = self.W[0]
        if self.idx == 1 or self.idx >= self.Lx:
            if len(self.sum_s)>1:
                self.sum_s = self.sum_s[:-1]
            if self.idx == self.Ln:
                if len(self.sum_n)>1:
                    self.sum_n = self.sum_n[:-1]
        self.N = np.concatenate(((noise_mask * Y)[None,], self.N[:-1,]), axis=0)
        self.X = np.concatenate(((speech_mask * Y)[None,], self.X[:-1,]), axis=0)
        self.Z = np.concatenate((Z[None,], self.Z[:-1]), axis=0)
        self.B = self.beta * self.B + np.einsum('...a,...b->...ab', noise_mask * Y,
                                                Y.conj())
        sum_n = np.maximum(np.sum(self.sum_n, axis=0), self.eps)
        B = self.B / sum_n
        # self.A = self.alpha * self.A + np.einsum('...a,...b->...ab',
        #                                                     speech_mask * Y, Y.conj())
        W = self.W
        # if np.sum(np.abs(self.P_z))<500:
        #     print(self.idx, np.sum(np.abs(self.P_z)))
        #     W = np.ones(W.shape, dtype=W.dtype)
        return np.expand_dims(W, axis=-1), B

class RLSEigenvectorTracking_withX():
    def __init__(self, channels=6, num_bins=513, alpha=0.99, beta=0.99, Lx=50, Ln=200,
                 eps=1e-10, sliding_window=True):
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        assert Lx <= Ln, 'The decaying speech window should be smaller or equal compared to the noise window'
        self.Lx = Lx
        self.Ln = Ln
        self.Q = np.array(
            [.01 * np.eye(channels, channels, dtype=np.complex128) for idx in
             range(num_bins)])
        self.W = np.ones((num_bins, channels), dtype=np.complex128)
        self.B = np.array(
            [.01 * np.eye(channels, channels, dtype=np.complex128) for idx in
             range(num_bins)])
        self.X = np.zeros((Lx, num_bins, channels), dtype=np.complex128)
        self.N = np.zeros((Ln, num_bins, channels), dtype=np.complex128)
        self.Z = np.zeros((Lx, num_bins), dtype=np.complex128)
        self.P_z =np.zeros((1,num_bins), dtype=np.complex128)
        self.r = np.zeros((1, num_bins, channels), dtype=np.complex128)
        self.num_bins = num_bins
        self.idx = 0
        self.sum_s = np.zeros((1, num_bins, 1, 1), dtype=np.complex128)
        self.sum_n = np.zeros((1, num_bins, 1, 1), dtype=np.complex128)
        self.sliding_window = sliding_window

    def __call__(self, Y, speech_mask, noise_mask):
        assert Y.ndim == 2, 'Only frequency and channel dim allowed'
        assert Y.shape[0] > Y.shape[
            1], 'The frequency dim is smaller than the channel dim, something must have gone wrong'
        F, D = Y.shape
        if noise_mask.ndim + 1 == Y.ndim:
            noise_mask = np.expand_dims(noise_mask, -1)
        if speech_mask.ndim + 1 == Y.ndim:
            speech_mask = np.expand_dims(speech_mask, -1)
        if self.idx == self.Ln:
            self.idx = self.idx
        elif self.idx >= self.Lx:
            self.idx += 1
        else:
            self.idx += 1

        if self.sliding_window:
            self.sum_s = np.concatenate((speech_mask[None,:,:,None], self.sum_s), axis=0)
            self.sum_n = np.concatenate((noise_mask[None,:,:,None], self.sum_n), axis=0)
            # step 1
            Z = np.einsum('...a,...a->...', self.W.conj(), speech_mask * Y)
            # Z = speech_mask * Y
            # step 2
            self.P_z = self.alpha * self.P_z +  Z * Z.conj() - self.alpha ** self.Lx * np.abs(
                self.Z[-1]) ** 2
            self.P_z[self.P_z==0] = self.eps
            P_z_inv = np.reciprocal(self.P_z)
            self.r = self.alpha * self.r + speech_mask * Y * np.expand_dims(
                Z,axis=-1).conj() - self.alpha ** self.Lx * self.X[-1] * np.expand_dims(
                self.Z[-1], axis=-1).conj()
            sum_s = np.sum(self.sum_s, axis=0)
            sum_s[sum_s==0] = self.eps
            r = self.r / (sum_s)
            # step 3
            a = np.tile([[self.beta, 0], [0, self.beta ** (1 - self.Ln)]],
                        [self.num_bins, 1, 1])
            N_N = np.stack([noise_mask * Y, self.N[-1]], axis=-1)
            b = np.linalg.inv(
                a + np.einsum('...bd, ...bc, ...ce->...de', N_N.conj(), self.Q, N_N))
            c = np.einsum('...ab,...bc,...cd->...ad', self.Q, N_N, b)
            d = np.einsum('...ab,...cb,...cd->...ad', c, N_N.conj(), self.Q)
            self.Q = (self.Q - d) / self.beta
            Q = self.Q * np.sum(self.sum_n, axis=0)
            # step 4
            self.W = np.einsum('...ab,...b->...a', Q, r)[0] * np.expand_dims(
                P_z_inv, axis=-1)
            self.W = self.W[0]
        else:
            self.sum_s =+ speech_mask[None,:,:,None]
            self.sum_n =+ noise_mask[None,:,:,None]
            # step 1
            Z = np.einsum('...a,...a->...', self.W.conj(), Y)
            # Z =  Y
            # step 2
            self.P_z = self.alpha * self.P_z + Z * Z.conj()
            self.P_z[self.P_z==0] = self.eps
            P_z_inv = np.reciprocal(self.P_z)
            self.r = self.alpha * self.r + Y * np.expand_dims(
                Z,axis=-1).conj()
            # sum_s = np.sum(self.sum_s, axis=0)
            r = self.r #/ (sum_s + self.eps)
            # step 3
            a = np.tile([self.beta],
                        [self.num_bins, 1, 1])
            N = (np.sqrt(noise_mask) * Y)[:,:,None]
            b = np.linalg.inv(
                a + np.einsum('...bd, ...bc, ...ce->...de', N.conj(), self.Q, N))
            c = np.einsum('...ab,...bc,...cd->...ad', self.Q, N, b)
            d = np.einsum('...ab,...cb,...cd->...ad', c, N.conj(), self.Q)
            self.Q = (self.Q - d) / self.beta
            Q = self.Q * np.sum(self.sum_n[0], axis=0)
            # step 4
            self.W = np.einsum('...ab,...b->...a', Q, r)[0] * np.expand_dims(
                P_z_inv, axis=-1)
            self.W = self.W[0]
        if self.idx == 1 or self.idx >= self.Lx:
            if len(self.sum_s)>1:
                self.sum_s = self.sum_s[:-1]
            if self.idx == self.Ln:
                if len(self.sum_n)>1:
                    self.sum_n = self.sum_n[:-1]
        self.N = np.concatenate(((noise_mask * Y)[None,], self.N[:-1,]), axis=0)
        self.X = np.concatenate(((speech_mask * Y)[None,], self.X[:-1,]), axis=0)
        self.Z = np.concatenate((Z[None,], self.Z[:-1]), axis=0)
        self.B = self.beta * self.B + np.einsum('...a,...b->...ab', noise_mask * Y,
                                                Y.conj())
        sum_n = np.maximum(np.sum(self.sum_n, axis=0), self.eps)
        B = self.B / sum_n
        # self.A = self.alpha * self.A + np.einsum('...a,...b->...ab',
        #                                                     speech_mask * Y, Y.conj())
        W = self.W
        # if np.sum(np.abs(self.P_z))<500:
        #     print(self.idx, np.sum(np.abs(self.P_z)))
        #     W = np.ones(W.shape, dtype=W.dtype)
        return np.expand_dims(W, axis=-1), B

class RLSEigenvectorTracking_MVDR():
    def __init__(self, A=None, B=None, channels=6, num_bins=513, alpha=0.99, beta=0.99, Lx=50, Ln=200,
                 eps=1e-10, sliding_window=True, souden=True):
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        assert Lx <= Ln, 'The decaying speech window should be smaller or equal compared to the noise window'
        self.Lx = Lx
        self.Ln = Ln

        self.Q = np.array(
            [.01 * np.eye(channels, channels, dtype=np.complex128) for _ in
             range(num_bins)])
        if A is None:
            self.A = np.array(
                [.01 * np.eye(channels, channels, dtype=np.complex128) for _ in
                 range(num_bins)])
        else:
            self.A = A
        # self.A = np.concatenate((np.ones((513,1,channels), dtype=np.complex128),
        #                          np.zeros((num_bins,channels-1,channels), dtype=np.complex128)),
        #                         axis=-2)
        if B is None:
            self.B = np.array(
                [.01 * np.eye(channels, channels, dtype=np.complex128) for idx in
                 range(num_bins)])
        else:
            self.B = B
        self.N = np.zeros((Ln, num_bins, channels), dtype=np.complex128)
        self.num_bins = num_bins
        self.idx = 0
        self.sum_s = np.zeros((1,num_bins, 1, 1), dtype=np.complex128)
        self.sum_n = np.zeros((1,num_bins, 1, 1), dtype=np.complex128)
        self.sliding_window = sliding_window
        self.souden = souden

    def __call__(self, Y, speech_mask, noise_mask):
        # assert Y.ndim == 2, 'Only frequency and channel dim allowed'
        assert Y.shape[-2] > Y.shape[-1], 'The frequency dim is smaller than the channel dim, something must have gone wrong'
        F, D = Y.shape
        Y = Y.squeeze()
        speech_mask = speech_mask.squeeze()
        noise_mask = noise_mask.squeeze()
        if noise_mask.ndim + 1 == Y.ndim:
            noise_mask = np.expand_dims(noise_mask, -1)
        if speech_mask.ndim + 1 == Y.ndim:
            speech_mask = np.expand_dims(speech_mask, -1)
        if self.idx == self.Ln:
            self.idx = self.idx
        elif self.idx >= self.Lx:
            self.idx += 1
        else:
            self.idx += 1
        self.sum_s = np.concatenate((speech_mask[None,:,:,None], self.sum_s), axis=0)
        self.sum_n = np.concatenate((noise_mask[None,:,:,None], self.sum_n), axis=0)
        A_tilde = np.einsum('...a,...b->...ab', speech_mask * Y, Y.conj())
        self.A = self.alpha * self.A + A_tilde
        # self.A = self.alpha * self.A + A_tilde -self.alpha**self.Lx * self.A_old[-1]
        sum_s = np.maximum(np.sum(self.sum_s, axis=0), self.eps)
        A = self.A / sum_s
        # step 3
        if self.sliding_window:
            a = np.tile([[self.beta, 0], [0, self.beta ** (1 - self.Ln)]],
                        [self.num_bins, 1, 1])
            N_N = np.stack([noise_mask * Y, self.N[-1]], axis=-1)
            b = np.array([np.linalg.inv(b) for b in (
                a + np.einsum('...bd, ...bc, ...ce->...de', N_N.conj(), self.Q, N_N))])
            c = np.einsum('...ab,...bc,...cd->...ad', self.Q, N_N, b)
            d = np.einsum('...ab,...cb,...cd->...ad', c, N_N.conj(), self.Q)
        else:
            a = np.tile([self.beta], [self.num_bins, 1, 1])
            N = (noise_mask * Y)[:,:,None]
            b = np.reciprocal(a + np.einsum('...bd, ...bc, ...ce->...de',
                                                   N.conj(), self.Q, N))
            c = np.einsum('...ab,...bc,...cd->...ad', self.Q, N, b)
            d = np.einsum('...ab,...cb,...cd->...ad', c, N.conj(), self.Q)
        self.Q = (self.Q - d) / self.beta
        Q = self.Q * np.sum(self.sum_n, axis=0)
        # step 4

        if self.souden:
            denominator = np.einsum('...ab,...bc->...ac', Q, A)
            nominator = np.trace(denominator, axis1=1, axis2=2)[:,None]
            W = denominator[:,:,0] / (nominator+self.eps)
        else:
            atf = get_pca_vector(A)
            numerator = np.einsum('...ab,...b->...a', Q, atf)
            denominator = np.einsum('...d,...d->...', atf.conj(), numerator)
            W = numerator / np.expand_dims(np.maximum(denominator,self.eps), axis=-1)

        if self.idx == 1 or self.idx >= self.Lx:
            self.sum_s = self.sum_s[:-1]
            if self.idx == self.Ln:
                self.sum_n = self.sum_n[1:]
        if self.sliding_window:
            # self.A_old = np.concatenate((A_tilde[None,], self.A_old[:-1, ]), axis=0)
            self.N = np.concatenate(((noise_mask * Y)[None,], self.N[:-1, ]),
                                    axis=0)

            self.B = self.beta * self.B + np.einsum('...a,...b->...ab', noise_mask * Y,
                                                    Y.conj())
        sum_n = np.maximum(np.sum(self.sum_n, axis=0), self.eps)
        B = self.B / (sum_n)


        # if np.sum(A_tilde)<100:
        #     # print(self.idx, np.sum(self.A))
        #     W = np.ones(W.shape, dtype=W.dtype)
        return np.expand_dims(W, axis=-1), B, A

def RLSEigenvectorTracking_MVDR_window(Y, speech_mask, noise_mask,
                                       Ln=200, Lx=50, eps=1e-10,
                                       alpha=0.95, beta=0.95, ban=True):
    # assert Y.ndim == 2, 'Only frequency and channel dim allowed'
    # assert Y.shape[-2] > Y.shape[-1], 'The frequency dim is smaller than the channel dim, something must have gone wrong'
    Y = Y.transpose(1,2,0)
    Y = Y.squeeze()
    speech_mask = np.real(speech_mask.squeeze())
    noise_mask = np.real(noise_mask.squeeze())
    if noise_mask.ndim + 1 == Y.ndim:
        noise_mask = np.expand_dims(noise_mask, -1)
    if speech_mask.ndim + 1 == Y.ndim:
        speech_mask = np.expand_dims(speech_mask, -1)
    sum_s = np.expand_dims(speech_mask[:Lx], -1)
    sum_n = np.expand_dims(noise_mask[:Lx], -1)
    A_tilde = np.einsum('...at,...bt->...ab', sum_s[:,:,0].transpose(1,2,0) * Y[:Lx].transpose(1,2,0), Y[:Lx].transpose(1,2,0).conj())
    A = A_tilde/np.maximum(np.sum(sum_s, keepdims=False, axis=0), eps)
    B_tilde = np.einsum('...at,...bt->...ab', sum_n[:,:,0].transpose(1,2,0) * Y[:Lx].transpose(1, 2, 0),
                  Y[:Lx].transpose(1, 2, 0).conj())
    num_bins = Y.shape[1]
    B = B_tilde/ np.maximum(
        np.sum(sum_n, keepdims=False, axis=0), eps)
    Q_tilde = np.array(
        [.01 * np.eye(Y.shape[-1], Y.shape[-1], dtype=np.complex128) for idx in
         range(num_bins)])
    for idx in range(Lx):
        a = np.tile([[beta, 0], [0, beta ** (1 - Ln)]],
                    [num_bins, 1, 1])
        N_N = np.stack([np.sqrt(noise_mask[idx]) * Y[idx], np.zeros((Y.shape[1], Y.shape[-1]), dtype=Y.dtype)], axis=-1)
        b = np.array([np.linalg.inv(np.maximum(b,eps)) for b in (
            a + np.einsum('...bd, ...bc, ...ce->...de', N_N.conj(), Q_tilde, N_N))])
        c = np.einsum('...ab,...bc,...cd->...ad', Q_tilde, N_N, b)
        d = np.einsum('...ab,...cb,...cd->...ad', c, N_N.conj(), Q_tilde)
        # print('Q', True if np.isnan(Q_tilde).any() else False)
        Q_tilde = (Q_tilde - d) / beta

    Y_out = np.zeros((Y.shape[0], Y.shape[1]), Y.dtype)

    N = np.zeros((200,Y.shape[1], Y.shape[-1]), dtype=Y.dtype)
    N[:Lx] = np.sqrt(np.squeeze(sum_n, axis=-1))*Y[:Lx]
    W = np.zeros(Y.shape, dtype=Y.dtype)
    # denominator = np.einsum('...ab,...bc->...ac', Q, A)
    # nominator = np.trace(denominator, axis1=1, axis2=2)[:, None]
    W_zw = get_mvdr_vector_souden(A,B) #(denominator[:, :, 0] / (nominator +eps))[None]
    if ban:
        W_zw = blind_analytic_normalization(W_zw, B)
    W[:Lx] = W_zw
    Y_out[:Lx] = np.einsum('...a,...a->...', W[:Lx].conj(),
                  Y[:Lx])
    for idx in range(Lx, Y.shape[0]):
        sum_s = np.concatenate((speech_mask[idx,:,:,None][None,], sum_s), axis=0)
        sum_n = np.concatenate((noise_mask[idx,:,:,None][None,], sum_n), axis=0)
        A_tilde = alpha * A_tilde + np.einsum('...a,...b->...ab', speech_mask[idx] * Y[idx], Y[idx].conj())
        # self.A = self.alpha * self.A + A_tilde -self.alpha**self.Lx * self.A_old[-1]
        sum_s_zw = np.maximum(np.sum(sum_s, axis=0), eps)
        A = A_tilde / sum_s_zw
        # step 3

        a = np.tile([[beta, 0], [0, beta ** (1 - Ln)]],
                    [num_bins, 1, 1])
        N_N = np.stack([np.sqrt(noise_mask[idx]) * Y[idx], N[-1]], axis=-1)
        b = np.array([np.linalg.inv(b) for b in (
            a + np.einsum('...bd, ...bc, ...ce->...de', N_N.conj(), Q_tilde, N_N))])
        c = np.einsum('...ab,...bc,...cd->...ad', Q_tilde, N_N, b)
        d = np.einsum('...ab,...cb,...cd->...ad', c, N_N.conj(), Q_tilde)
        # print('Q', True if np.isnan(Q_tilde).any() else False)
        Q_tilde = (Q_tilde - d) / beta
        Q = Q_tilde * np.sum(sum_n, axis=0)
        # step 4
        denominator = np.einsum('...ab,...bc->...ac', Q, A)
        nominator = np.trace(denominator, axis1=1, axis2=2)[:,None]
        W[idx] = denominator[:,:,0] / (nominator + eps)
        if ban:
            W[idx, :, :] = blind_analytic_normalization(W[idx, :, :], B)

        B_tilde = beta * B_tilde + np.einsum('...a,...b->...ab', noise_mask[idx] * Y[idx],
                                                Y[idx].conj())
        sum_n_zw = np.maximum(np.sum(sum_n, axis=0), eps)
        B = B_tilde / (sum_n_zw)
        sum_s = sum_s[:-1]
        Y_out[idx] = np.einsum('...a,...a->...', W[idx].conj(),
                  Y[idx])
        if idx >= Ln-Lx:
            sum_n = sum_n[1:]
        N = np.concatenate(((np.sqrt(noise_mask[idx]) * Y[idx])[None,], N[:-1, ]),
                            axis=0)
        # print('W', True if np.isnan(W).any() else False)
        # print('Y', True if np.isnan(Y_out).any() else False)

    # if np.sum(A_tilde)<100:
    #     # print(self.idx, np.sum(self.A))
    #     W = np.ones(W.shape, dtype=W.dtype)
    return Y_out

class MVDR_online():
    def __init__(self, channels=6, num_bins=513, alpha=0.99, beta=0.99, Lx=25, Ln=100,
                 eps=1e-10):
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        assert Lx <= Ln, 'The decaying speech window should be smaller or equal compared to the noise window'
        self.A_tilde = np.array(
            [.01 * np.eye(channels, channels, dtype=np.complex128) for idx in
             range(num_bins)])
        self.B_tilde = np.array(
            [.01 * np.eye(channels, channels, dtype=np.complex128) for idx in
             range(num_bins)])

        self.num_bins = num_bins
        self.idx = 0
        self.sum_s = np.zeros((num_bins, 1, 1), dtype=np.complex128)
        self.sum_n = np.zeros((num_bins, 1, 1), dtype=np.complex128)

    def __call__(self, Y, speech_mask, noise_mask):
        assert Y.shape[-2] > Y.shape[-1],\
            'The frequency dim is smaller than the channel dim,' \
            ' something must have gone wrong'

        if Y.ndim == 2:
            F, D = Y.shape
        elif Y.ndim == 3:
            T, F, D = Y.shape
        else: raise ValueError('Y.ndim = {} ist to large,'
                               ' only 2 or 3 dimensions are allowed'.format(Y.ndim))
        if speech_mask.ndim + 1 == Y.ndim:
            speech_mask = np.expand_dims(speech_mask, -1)
        if noise_mask.ndim + 1 == Y.ndim:
            noise_mask = np.expand_dims(noise_mask, -1)
        A_tilde = self.alpha * self.A_tilde + np.sum(np.einsum('...d,...e->...de',
                                                                 speech_mask * Y, Y.conj()), axis=0)
        B_tilde = self.beta * self.B_tilde + np.sum(np.einsum('...d,...e->...de',
                                                                 noise_mask * Y, Y.conj()), axis=0)

        self.sum_s += np.expand_dims(speech_mask, axis=-1)
        self.sum_n += np.expand_dims(noise_mask, axis=-1)
        # sum_s = self.sum_s
        # sum_s[sum_s==0] = self.eps
        # sum_n = self.sum_n
        # sum_n[sum_n==0]  = self.eps
        A = A_tilde/np.maximum(self.sum_s,self.eps)
        B = B_tilde
        Q = np.linalg.inv(B)*self.sum_n
        denominator = np.einsum('...ab,...bc->...ac', Q, A)
        nominator = np.trace(denominator, axis1=1, axis2=2)[:,None]
        # nominator[nominator==0] = self.eps
        W = denominator[:,:,0] / (nominator + self.eps)
        self.A_tilde = A_tilde
        self.B_tilde = B_tilde
        self.B = B_tilde/np.maximum(self.sum_s,self.eps)
        if np.sum(A)<100:
            # print(self.idx, np.sum(self.A))
            W = np.ones(W.shape, dtype=W.dtype)
        return np.expand_dims(W, axis=-1), B

class MVDR_online2():
    def __init__(self, channels=6, num_bins=513, alpha=0.99, beta=0.99, Lx=25, Ln=100,
                 eps=1e-10):
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        assert Lx <= Ln, 'The decaying speech window should be smaller or equal compared to the noise window'
        self.A_tilde = np.array(
            [.01 * np.eye(channels, channels, dtype=np.complex128) for idx in
             range(num_bins)])
        self.B_tilde = np.array(
            [.01 * np.eye(channels, channels, dtype=np.complex128) for idx in
             range(num_bins)])

        self.num_bins = num_bins
        self.idx = 0
        self.sum_s = np.zeros((num_bins, 1, 1), dtype=np.complex128)
        self.sum_n = np.zeros((num_bins, 1, 1), dtype=np.complex128)

    def __call__(self, Y, speech_mask, noise_mask):
        assert Y.shape[-2] > Y.shape[-1],\
            'The frequency dim is smaller than the channel dim,' \
            ' something must have gone wrong'

        if Y.ndim == 2:
            F, D = Y.shape
        elif Y.ndim == 3:
            T, F, D = Y.shape
        else: raise ValueError('Y.ndim = {} ist to large,'
                               ' only 2 or 3 dimensions are allowed'.format(Y.ndim))
        if speech_mask.ndim + 1 == Y.ndim:
            speech_mask = np.expand_dims(speech_mask, -1)
        if noise_mask.ndim + 1 == Y.ndim:
            noise_mask = np.expand_dims(noise_mask, -1)
        A_tilde = self.A_tilde + np.sum(np.einsum('...d,...e->...de',
                                                                 speech_mask * Y, Y.conj()), axis=0)
        B_tilde = self.B_tilde + np.sum(np.einsum('...d,...e->...de',
                                                                 noise_mask * Y, Y.conj()), axis=0)

        A = A_tilde
        B = B_tilde
        Q = np.linalg.inv(B)
        denominator = np.einsum('...ab,...bc->...ac', Q, A)
        nominator = np.trace(denominator, axis1=1, axis2=2)[:,None]
        # nominator[nominator==0] = self.eps
        W = denominator[:,:,0] / np.complex(np.maximum(np.real(nominator), self.eps), np.imag(nominator))
        self.A_tilde = A_tilde
        self.B_tilde = B_tilde
        self.B = B_tilde/np.maximum(self.sum_n,self.eps)
        return np.expand_dims(W, axis=-1), B


def interpolating_beamforming_vector(w, speech_mask, K=3, cluster='mean'):
    speech_mask_ext = np.zeros(w.shape, dtype=w.dtype)
    speech_mask_arg = np.zeros(speech_mask.shape, dtype=speech_mask.dtype)
    def get_max(mask):
        return np.repeat(
            np.array([np.max(speech_mask[idy * K:idy * K + K], axis=-1, keepdims=False)
                      for idy in range(int(np.ceil(mask.shape[-1] / K)))]), K, axis=-1)
    def get_arg(mask):
        return np.repeat(np.array([np.argmax(mask[idy * K:idy * K + K], axis=-1)
                                   for idy in range(int(np.ceil(mask.shape[-1] / K)))]).T,
                         K, axis=-1)
    #
    w_new = np.zeros(w.shape, dtype=w.dtype)
    if cluster=='mean':
        w_zw = np.repeat(np.array([np.mean(w[idx * K: idx * K + K], axis=0) for idx in
                         range(int(np.ceil(w.shape[-2] / K)))]), K, axis=-2)
    elif cluster=='mask':
        max_mask = get_max(speech_mask)
        min_mask = get_max(1 - speech_mask)
        arg_mask = np.argmax((min_mask, max_mask), axis=0)
        arg_max_mask = get_arg(speech_mask)
        arg_min_mask = get_arg(1 - speech_mask)
        speech_mask_arg[arg_mask == 1] = arg_max_mask[arg_mask == 1]
        speech_mask_arg[arg_mask == 0] = arg_min_mask[arg_mask == 0]
        speech_mask_ext[arg_mask == 1, :] = w[arg_mask == 1, :]
        speech_mask_ext[arg_mask == 0, :] = w[arg_mask == 0, :]
        w_zw = speech_mask_ext
    else:
        return w
    for l in range(int(np.ceil(w.shape[-2]/K))):
        for k in range(K):
            # if not l*K+k >= w.shape[-2]:
            if cluster=='mean':
                c=np.abs(1-k)/K
            elif cluster=='mask':
                c = 1 - np.abs(speech_mask_arg[l * K + k, None] - k) / K
            if (l + 1)*K >= w_zw.shape[-2]:
                w_new[l*K+k,:]=(1-c)*w_zw[l*K,:]+c*w_zw[-1,:]
            else:
                w_new[l*K+k,:]=(1-c)*w_zw[l*K,:]+c*w_zw[(l+1)*K,:]
    return (w_new/np.maximum(np.abs(w_new),1e-6)).squeeze()

def interpolate_masks(speech_mask, K=5):
    # speech_mask_ext = np.zeros((speech_mask.shape[0],int(np.ceil(speech_mask.shape[-1]/K))), dtype=speech_mask.dtype)
    speech_mask_ext = np.zeros(speech_mask.shape, dtype = speech_mask.dtype)
    speech_mask_arg = np.zeros((speech_mask.shape[0],int(np.ceil(speech_mask.shape[-1]/K))), dtype=speech_mask.dtype)
    for idy in range(int(np.ceil(speech_mask.shape[-1]/K))):
        if np.max(speech_mask[idy*K:idy*K+K], axis=-1) > 1-np.min(speech_mask[idy*K:idy*K+K], axis=-1):
    #         speech_mask_ext[:,idy*K:idy*K+K] = np.max(speech_mask[:,idy*K:idy*K+K], axis=-1)
            speech_mask_arg[idy] = np.argmax(speech_mask[idy*K:idy*K+K], axis=-1)
            speech_mask_ext[idy*K:idy*K+K] = speech_mask[speech_mask_arg[idy]]
        else:
    #         speech_mask_ext[:,idy*K:idy*K+K] = np.min(speech_mask[:,idy*K:idy*K+K], axis=-1)
            speech_mask_arg[idy] = np.argmin(speech_mask[idy*K:idy*K+K], axis=-1)
            speech_mask_ext[idy*K:idy*K+K] = speech_mask[speech_mask_arg[idy]]
    return speech_mask_ext

