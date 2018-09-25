""" Beamformer module.

The shape convention is to place time at the end to speed up computation and
move independent dimensions to the front.

That results i.e. in the following possible shapes:
    X: Shape (F, D, T).
    mask: Shape (F, K, T).
    PSD: Shape (F, D, D).

# TODO: These shape hints do not fit together. If mask has K, PSD needs it, too.

The functions themselves are written more generic, though.
"""

import warnings
from scipy.linalg import sqrtm
import numpy as np
from numpy.linalg import solve
from scipy.linalg import eig
from scipy.linalg import eigh
from nt.math.correlation import covariance  # as shortcut!
from nt.math.solve import stable_solve


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
    normalization = np.divide(  # https://stackoverflow.com/a/37977222/5766934
        nominator, denominator,
        out=np.zeros_like(nominator),
        where=denominator != 0
    )
    return vector * np.abs(normalization[..., np.newaxis])


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


def get_mvdr_vector_souden_old(
        target_psd_matrix,
        noise_psd_matrix,
        ref_channel=0,
        eps=1e-5,
):
    """
    Returns the MVDR beamforming vector described in [Souden10].

    :param target_psd_matrix: Target PSD matrix
        with shape (..., bins, sensors, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (..., bins, sensors, sensors)
    :param ref_channel:
    :param eps:
    :return: Set of beamforming vectors with shape (..., bins, sensors)
    """

    # Make sure matrix is hermitian
    shape = target_psd_matrix.shape
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
                f, target_psd_matrix[f, :, :],
                noise_psd_matrix[f, :, :]))
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError('Error for frequency {}\n'
                                        'phi_xx: {}\n'
                                        'phi_nn: {}'.format(
                f, target_psd_matrix[f, :, :],
                noise_psd_matrix[f, :, :]))
    denominator = np.trace(numerator, axis1=-1, axis2=-2)
    beamforming_vector = numerator[:, ref_channel] / np.expand_dims(denominator + eps, axis=-1)
    beamforming_vector = beamforming_vector.reshape(shape[:-1])
    return beamforming_vector


def get_mvdr_vector_souden(
        target_psd_matrix,
        noise_psd_matrix,
        ref_channel=None,
        eps=None,
        return_ref_channel=False
):
    """
    Returns the MVDR beamforming vector described in [Souden2010MVDR].
    The implementation is based on the description of [Erdogan2016MVDR].

    The ref_channel is selected based of an SNR estimate.

    The eps ensures that the SNR estimation for the ref_channel works
    as long target_psd_matrix and noise_psd_matrix do not contain inf or nan.
    Also zero matrices work. The default eps is the smallest non zero value.

    Note: the frequency dimension is necessary for the ref_channel estimation.
    Note: Currently this function does not support independent dimensions with
          an estimated ref_channel. There is an open point to discuss:
          Should the independent dimension be considered in the SNR estimate
          or not?

    :param target_psd_matrix: Target PSD matrix
        with shape (..., bins, sensors, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (..., bins, sensors, sensors)
    :param ref_channel:
    :param return_ref_channel:
    :param eps: If None use the smallest number bigger than zero.
    :return: Set of beamforming vectors with shape (bins, sensors)

    Returns:

    @article{Souden2010MVDR,
      title={On optimal frequency-domain multichannel linear filtering for noise reduction},
      author={Souden, Mehrez and Benesty, Jacob and Affes, Sofi{\`e}ne},
      journal={IEEE Transactions on audio, speech, and language processing},
      volume={18},
      number={2},
      pages={260--276},
      year={2010},
      publisher={IEEE}
    }
    @inproceedings{Erdogan2016MVDR,
      title={Improved MVDR Beamforming Using Single-Channel Mask Prediction Networks.},
      author={Erdogan, Hakan and Hershey, John R and Watanabe, Shinji and Mandel, Michael I and Le Roux, Jonathan},
      booktitle={Interspeech},
      pages={1981--1985},
      year={2016}
    }

    """
    phi = stable_solve(noise_psd_matrix, target_psd_matrix)
    lambda_ = np.trace(phi, axis1=-1, axis2=-2)[..., None, None]
    if eps is None:
        eps = np.finfo(lambda_.dtype).tiny
    mat = phi / np.maximum(lambda_.real, eps)
    
    if ref_channel is None:
        if phi.ndim != 3:
            raise ValueError(
                'Estimating the ref_channel expects currently that the input '
                'has 3 ndims (frequency x sensors x sensors). '
                'Considering an independent dim in the SNR estimate is not '
                'unique.'
            )
        SNR = np.einsum(
            '...FdR,...FdD,...FDR->...R', mat.conj(), target_psd_matrix, mat
        ) / np.maximum(np.einsum(
            '...FdR,...FdD,...FDR->...R', mat.conj(), noise_psd_matrix, mat
        ), eps)
        # Raises an exception when np.inf and/or np.NaN was in target_psd_matrix
        # or noise_psd_matrix
        assert np.all(np.isfinite(SNR)), SNR
        ref_channel = np.argmax(SNR.real)

    assert np.isscalar(ref_channel), ref_channel
    beamformer = mat[..., ref_channel]

    if return_ref_channel:
        return beamformer, ref_channel
    else:
        return beamformer


def get_lcmv_vector_souden(
        target_psd_matrix,
        interference_psd_matrix,
        noise_psd_matrix,
        ref_channel=None,
        eps=None,
        return_ref_channel=False
):
    """
    In "A Study of the LCMV and MVDR Noise Reduction Filters" Mehrez Souden
    elaborates an alternative formulation for the LCMV beamformer in the
    appendix for a rank one interference matrix.

    Therefore, this algorithm is only valid, when the interference PSD matrix
    is approximately rank one, or (in other words) only 2 speakers are present
    in total.

    Args:
        target_psd_matrix:
        interference_psd_matrix:
        noise_psd_matrix:
        ref_channel:
        eps:
        return_ref_channel:

    Returns:

    """
    phi_in = stable_solve(noise_psd_matrix, interference_psd_matrix)
    phi_xn = stable_solve(noise_psd_matrix, target_psd_matrix)

    D = phi_in.shape[-1]

    # Equation 5, 6
    gamma_in = np.trace(phi_in, axis1=-1, axis2=-2)[..., None, None]
    gamma_xn = np.trace(phi_xn, axis1=-1, axis2=-2)[..., None, None]

    # Can be written in a single einsum call, here separate for clarity
    # Equation 11
    gamma = gamma_in * gamma_xn - np.trace(
        np.einsum('...ab,...bc->...ac', phi_in, phi_xn)
    )[..., None, None]
    # Possibly:
    # gamma = gamma_in * gamma_xn - np.einsum('...ab,...ba->...', phi_in, phi_xn)

    eye = np.eye(D)[(phi_in.ndim - 2) * [None] + [...]]

    # TODO: Should be determined automatically (per speaker)?
    ref_channel = 0

    # Equation 51, first fraction
    if eps is None:
        eps = np.finfo(gamma.dtype).tiny
    mat = gamma_in * eye - phi_in / np.maximum(gamma.real, eps)

    # Equation 51
    # Faster, when we select the ref_channel before matrix multiplication.
    beamformer = np.einsum('...ab,...bc->...ac', mat, phi_xn)[..., ref_channel]
    # beamformer = np.einsum('...ab,...b->...a', mat, phi_xn[..., ref_channel])

    if return_ref_channel:
        return beamformer, ref_channel
    else:
        return beamformer
