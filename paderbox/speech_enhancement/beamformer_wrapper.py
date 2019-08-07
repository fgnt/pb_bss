import numpy as np
from typing import Optional

import paderbox as pb
from .beamformer import *
from paderbox.utils.numpy_utils import morph
from paderbox.math.correlation import covariance
from scipy.signal import lfilter


def get_pca_rank_one_estimate(covariance_matrix, **atf_kwargs):
    """
    Estimates the matrix as the outer product of the dominant eigenvector.
    """
    # Calculate eigenvals/vecs
    a = get_pca_vector(covariance_matrix, **atf_kwargs)
    cov_rank1 = np.einsum('...d,...D->...dD', a, a.conj())
    scale = np.trace(covariance_matrix, axis1=-1, axis2=-2) / np.trace(
        cov_rank1, axis1=-1, axis2=-2)
    return scale[..., None, None] * cov_rank1


def _get_gev_atf_vector(
        covariance_matrix,
        noise_covariance_matrix,
        **gev_kwargs
):
    """Get the dominant generalized eigenvector as an ATF estimate.

    [1] https://arxiv.org/pdf/1707.00201.pdf
    """
    # [1] Equation (27)
    w = get_gev_vector(
        covariance_matrix,
        noise_covariance_matrix,
        **gev_kwargs
    )

    # [1] Equation (27)
    return np.einsum('...dD,...D->...d', noise_covariance_matrix, w)


def get_gev_rank_one_estimate(
        covariance_matrix,
        noise_covariance_matrix,
        **gev_kwargs,
):
    """
    Estimates the matrix as the outer product of the generalized eigenvector.
    """
    a = _get_gev_atf_vector(
        covariance_matrix, noise_covariance_matrix, **gev_kwargs
    )
    cov_rank1 = np.einsum('...d,...D->...dD', a, a.conj())
    scale = np.trace(covariance_matrix, axis1=-1, axis2=-2)
    scale /= np.trace(cov_rank1, axis1=-1, axis2=-2)
    return scale[..., None, None] * cov_rank1


def _get_atf_vector(
        atf_type,
        target_psd_matrix,
        noise_psd_matrix,
        **atf_kwargs
):
    if atf_type == 'pca':
        return get_pca_vector(target_psd_matrix, **atf_kwargs)
    elif atf_type == 'scaled_gev_atf':
        # this atf type is called scaled_gev_atf to clarify that it is not
        # a gev beamforming vector but a scaled atf estimated using the same
        # projection used in the GEV Cholesky decomposition
        return _get_gev_atf_vector(
            target_psd_matrix,
            noise_psd_matrix,
            **atf_kwargs,
        )
    else:
        raise ValueError(atf_type, 'use either pca or scaled_gev_atf')


def _get_rank_1_approximation(
        atf_type,
        target_psd_matrix,
        noise_psd_matrix,
        **atf_kwargs
):
    if atf_type == 'rank1_pca':
        return get_pca_rank_one_estimate(target_psd_matrix, **atf_kwargs)
    elif atf_type == 'rank1_gev':
        return get_gev_rank_one_estimate(
            target_psd_matrix, noise_psd_matrix, **atf_kwargs)
    else:
        raise ValueError(atf_type, 'use either rank1_pca or rank1_gev')


def _get_response_vector(source_index, num_sources, epsilon=0.):
        response_vector = pb.utils.numpy_utils.labels_to_one_hot(
            np.array(source_index),
            num_sources,
            dtype=np.float64
        )
        response_vector = np.clip(response_vector, epsilon, 1.)
        return response_vector


def get_bf_vector(
        beamformer,
        target_psd_matrix,
        noise_psd_matrix=None,
        **bf_kwargs
):
    """
    # ToDo: how do we use the lcmv beamformer in this context?
     Wrapper for all beamformer
    Args:
        beamformer: string defining the kind of beamforming vector.
            Different steps of the beamforming vector estimation have to be
            separated with a ´+´ e.g. ´rank1_gev+mvdr_souden+ban´
        target_psd_matrix: `Array` of shape (..., sensor, sensor)
            with the covariance statistics for the target signal.
        noise_psd_matrix: `Array` of shape (..., sensor, sensor)
            with the covariance statistics for the interference signal.
        **bf_kwargs: option for the beamformer estimation
            if necessary, options for atf vector estimation may be added to
            the bf_kwargs under the key atf_kwargs. If no atf kwargs are
            added the code falls back to the defaults.

    Returns: beamforming vector

    """
    assert 'lcmv' not in beamformer, (
        'Since the LCMV beamformer and its variants sufficiently differ from '
        'all other beamforming approaches, we provide a separate wrapper '
        'function `get_multi_source_bf_vector()`.'
    )
    assert isinstance(beamformer, str), beamformer

    if beamformer.endswith('+ban'):
        ban = True
        beamformer_core = beamformer[:-len('+ban')]
    else:
        ban = False
        beamformer_core = beamformer

    if beamformer_core == 'pca':
        beamforming_vector = get_pca_vector(target_psd_matrix, **bf_kwargs)
    elif beamformer_core in ['pca+mvdr', 'scaled_gev_atf+mvdr']:
        atf, _ = beamformer_core.split('+')
        atf_vector = _get_atf_vector(
            atf, target_psd_matrix,
            noise_psd_matrix,
            **bf_kwargs.pop('atf_kwargs', {})
        )
        beamforming_vector = get_mvdr_vector(atf_vector, noise_psd_matrix)
    elif beamformer_core in [
        'mvdr_souden',
        'rank1_pca+mvdr_souden',
        'rank1_gev+mvdr_souden',
    ]:
        if not beamformer_core == 'mvdr_souden':
            rank1_type, _ = beamformer_core.split('+')
            target_psd_matrix = _get_rank_1_approximation(
                rank1_type,
                target_psd_matrix,
                noise_psd_matrix,
                **bf_kwargs.pop('atf_kwargs', {})
            )
        beamforming_vector = get_mvdr_vector_souden(
            target_psd_matrix,
            noise_psd_matrix,
            **bf_kwargs,
        )
    elif beamformer_core in ['gev', 'rank1_pca+gev', 'rank1_gev+gev']:
        # rank1_gev+gev is not supported since it should no differ from gev
        if not beamformer_core == 'gev':
            rank1_type, _ = beamformer_core.split('+')
            target_psd_matrix = _get_rank_1_approximation(
                rank1_type,
                target_psd_matrix,
                noise_psd_matrix,
                **bf_kwargs.pop('atf_kwargs', {})
            )
        beamforming_vector = get_gev_vector(
            target_psd_matrix,
            noise_psd_matrix,
            **bf_kwargs,
        )
    elif beamformer_core in ['wmwf', 'rank1_pca+wmwf', 'rank1_gev+wmwf']:
        if not beamformer_core == 'wmwf':
            rank1_type, _ = beamformer_core.split('+')
            target_psd_matrix = _get_rank_1_approximation(
                rank1_type,
                target_psd_matrix,
                noise_psd_matrix,
                **bf_kwargs.pop('atf_kwargs', {})
            )
        beamforming_vector = get_wmwf_vector(
            target_psd_matrix,
            noise_psd_matrix,
            **bf_kwargs,
        )
    else:
        raise ValueError(
            f'Could not find implementation for {beamformer_core}.\n'
            f'Original call contained {beamformer}.'
        )

    if ban:
        beamforming_vector = blind_analytic_normalization(
            beamforming_vector,
            noise_psd_matrix
        )

    return beamforming_vector


def get_multi_source_bf_vector(
        beamformer: str,
        target_psd_matrix: np.array,
        interference_psd_matrix: np.array,
        noise_psd_matrix: np.array,
        source_index: int,
        epsilon: float = 0.,
        *,
        denominator_matrix_for_atf: Optional[str]=None,
        denominator_matrix_for_bf: str,
        denominator_matrix_for_ban: Optional[str]=None,
        **bf_kwargs
):
    """Wrapper for LCMV and friends.

    This wrapper has a similar interface as `get_bf_vector()`.

    Args:
        beamformer:
        target_psd_matrix: Shape (K, F, D, D)
        interference_psd_matrix: Shape (K, F, D, D)
        noise_psd_matrix: Shape (F, D, D)
        source_index: Int in {0, ... K - 1}.
        epsilon: Sharon Gannot recommends values larger than zero to avoid
            over-suppression of the interference speaker. You may want to
            compare this to zero-forcing equalizer in our EDK lectures.
        denominator_matrix_for_atf: Either 'noise' or 'interference'.
        denominator_matrix_for_bf: Either 'noise' or 'interference'.
        denominator_matrix_for_ban: Either 'noise' or 'interference'.
        **bf_kwargs:

    Returns:

    """
    k = source_index
    K = target_psd_matrix.shape[0]

    assert isinstance(beamformer, str), beamformer

    if beamformer.endswith('+ban'):
        ban = True
        beamformer_core = beamformer[:-len('+ban')]
    else:
        ban = False
        beamformer_core = beamformer

    if beamformer_core in ['pca+lcmv', 'scaled_gev_atf+lcmv']:
        if beamformer_core == 'pca+lcmv':
            assert denominator_matrix_for_atf is None, \
                denominator_matrix_for_atf
        else:
            if denominator_matrix_for_atf == 'noise':
                denominator_matrix_for_atf \
                    = np.repeat(noise_psd_matrix[None, :, :, :], K, axis=0)
            elif denominator_matrix_for_atf == 'interference':
                denominator_matrix_for_atf = interference_psd_matrix
            else:
                raise ValueError(denominator_matrix_for_atf)

        atf, _ = beamformer_core.split('+')
        atf_vector = _get_atf_vector(
            atf,
            target_psd_matrix,
            denominator_matrix_for_atf,
            **bf_kwargs.pop('atf_kwargs', {})
        )

        if denominator_matrix_for_bf == 'noise':
            denominator_matrix_for_bf = noise_psd_matrix
        elif denominator_matrix_for_bf == 'interference':
            denominator_matrix_for_bf = interference_psd_matrix[k, :, :, :]
        else:
            raise ValueError(denominator_matrix_for_bf)

        response_vector = _get_response_vector(
            source_index=source_index,
            num_sources=target_psd_matrix.shape[0],
            epsilon=epsilon,
        )

        beamforming_vector = get_lcmv_vector(
            atf_vectors=atf_vector,
            response_vector=response_vector,
            noise_psd_matrix=denominator_matrix_for_bf,
        )
    elif beamformer_core in [
        'lcmv_souden',
        'rank1_pca+lcmv_souden',
        'rank1_gev+lcmv_souden',
    ]:
        raise NotImplementedError(
            f'All Souden LCMV variants not yet implemented: {beamformer_core}'
        )
    else:
        raise ValueError(
            f'Could not find implementation for {beamformer_core}.\n'
            f'Original call contained {beamformer}.'
        )

    if ban:
        if denominator_matrix_for_ban == 'noise':
            denominator_matrix_for_ban = noise_psd_matrix
        elif denominator_matrix_for_ban == 'interference':
            denominator_matrix_for_ban = interference_psd_matrix[k, :, :, :]
        else:
            raise ValueError(denominator_matrix_for_ban)

        beamforming_vector = blind_analytic_normalization(
            beamforming_vector,
            denominator_matrix_for_ban,
        )
    else:
        assert denominator_matrix_for_ban is None, denominator_matrix_for_ban

    return beamforming_vector


def get_multi_source_bf_vector_from_masks(
        observation_stft,
        mask,
        method,
        lcmv_denominator_matrix_for_atf,
        lcmv_denominator_matrix_for_bf,
        lcmv_denominator_matrix_for_ban,
        lcmv_epsilon=None,
):
    """

    Args:
        observation_stft: Shape (F, T, D)
        mask: Shape (F, K, T)
        method: Strings. See `get_bf_vector()`.
        lcmv_denominator_matrix_for_atf:
        lcmv_denominator_matrix_for_bf:
        lcmv_denominator_matrix_for_ban:
        lcmv_epsilon: None or float scalar

    Returns: Beamforming vector with shape (F, K, D)

    """
    F, T, D = observation_stft.shape
    _, K, _ = mask.shape
    np.testing.assert_equal(mask.shape[0], observation_stft.shape[0])
    np.testing.assert_equal(mask.shape[2], observation_stft.shape[1])

    # Sums up all other masks (all masks but mask k).
    interference_mask = np.stack(
        [np.sum(np.delete(mask, k, axis=1), axis=1) for k in range(K)],
        axis=1
    )
    np.testing.assert_equal(mask.shape, interference_mask.shape)

    mask = np.clip(mask, 1e-10, 1)
    interference_mask = np.clip(interference_mask, 1e-10, 1)

    target_psd \
        = pb.speech_enhancement.beamformer.get_power_spectral_density_matrix(
            morph('ftd->fdt', observation_stft),
            morph('fkt->fkt', mask)
        )
    interference_psd \
        = pb.speech_enhancement.beamformer.get_power_spectral_density_matrix(
            morph('ftd->fdt', observation_stft),
            morph('fkt->fkt', interference_mask)
        )
    np.testing.assert_equal(target_psd.shape, (F, K, D, D))
    np.testing.assert_equal(interference_psd.shape, (F, K, D, D))

    if 'lcmv' in method.split('+'):
        # Assumes that the noise class is the last one.
        beamforming_vector = np.stack(list(
            pb.speech_enhancement.get_multi_source_bf_vector(
                method,
                target_psd_matrix=morph('fkdD->kfdD', target_psd)[:K - 1],
                interference_psd_matrix=morph('fkdD->kfdD', interference_psd)[:K - 1],
                noise_psd_matrix=target_psd[:, -1, :, :],
                source_index=k,
                epsilon=lcmv_epsilon,
                denominator_matrix_for_atf=lcmv_denominator_matrix_for_atf,
                denominator_matrix_for_bf=lcmv_denominator_matrix_for_bf,
                denominator_matrix_for_ban=lcmv_denominator_matrix_for_ban,
            ) for k in range(K - 1)
        ), axis=1)

        value = np.max(np.abs(beamforming_vector))
        if value > 1e10:
            import warnings
            warnings.warn(
                'You seem to have encountered instabilities: '
                f'np.max(np.abs(beamforming_vector)) = {value}'
            )
    else:
        beamforming_vector = np.stack(list(
            pb.speech_enhancement.get_single_source_bf_vector(
                method,
                target_psd_matrix=target_psd[:, k, :, :],
                noise_psd_matrix=interference_psd[:, k, :, :],
            ) for k in range(K)
        ), axis=1)

    return beamforming_vector


def block_online_beamforming(
        observation,
        target_mask,
        noise_mask,
        *,
        block_size=5,
        target_psd_init=None,
        noise_psd_init=None,
        beamformer='mvdr_souden',
        target_decay_factor=0.95,
        noise_decay_factor=0.95,
        noise_psd_normalization=False,
        eps=1e-10,
        return_bf_vector=False,
        **beamformer_kwargs
):

    """
    :param observation: Observed signal
        with shape (..., bins, sensors, frames)
    :param target_mask: Target mask
        with shape (..., bins, frames)
    :param noise_mask: Noise mask
        with shape(..., bins, frames)
    :param block_size:
    :param target_psd_init: Target PSD matrix initalization
        with shape (..., bins, sensors, sensors)
    :param noise_psd_init: Noise PSD matrix initialization
        with shape (..., bins, sensors, sensors)
    :param beamformer: name of the beamformer used
    :param target_decay_factor:
    :param noise_decay_factor:
    :param noise_psd_normalization:
    :param eps:
    :param return_bf_vector:
    :param beamformer_kwargs:
    :return:
    """
    # split the inputs to segments of block_size
    shape = observation.shape
    ndims = observation.ndim
    assert len(shape) >= 3
    padding = (observation.ndim - 1) * [[0, 0]] + [
        [0, block_size - shape[-1] % block_size]]
    observation = np.pad(observation, padding, 'constant')
    observation = morph('...t*b->t...b', observation, b=block_size)
    target_mask = np.pad(target_mask, padding[1:], 'constant')
    target_mask = morph('...t*b->t...b', target_mask, b=block_size)
    noise_mask = np.pad(noise_mask, padding[1:], 'constant')
    noise_mask = morph('...t*b->t...b', noise_mask, b=block_size)
    target_psd = covariance(observation, target_mask, normalize=False,
                            force_hermitian=True)
    noise_psd = covariance(observation, noise_mask,
                           normalize=noise_psd_normalization)
    if target_psd_init is None:
        target_psd_init = np.zeros_like(target_psd[0])
    if noise_psd_init is None:
        noise_psd_init = np.zeros_like(noise_psd[0])
        noise_psd_init += np.reshape(
            eps * np.eye(shape[-2], dtype=noise_psd.dtype),
            (ndims-2) * [1] + [shape[-2], shape[-2]]
        )
    updated_target_psd = np.concatenate(
        [target_psd_init[None], target_psd], axis=0
    )
    updated_noise_psd = np.concatenate(
        [noise_psd_init[None] / (1 - noise_decay_factor), noise_psd], axis=0
    )
    unbiased_noise_psd = lfilter([noise_decay_factor],
                                 [1., -noise_decay_factor],
                                 updated_noise_psd, axis=0)[1:]
    unbiased_targed_psd = lfilter([target_decay_factor],
                                  [1., -target_decay_factor],
                                  updated_target_psd, axis=0)[1:]
    unbiased_noise_psd = unbiased_noise_psd / (
        1-noise_decay_factor**np.reshape(
            np.arange(1, noise_psd.shape[0]+1), ([-1] + ndims*[1])
        ))
    unbiased_targed_psd = unbiased_targed_psd / (
        1 - target_decay_factor ** np.reshape(
            np.arange(1, target_psd.shape[0]+1), ([-1] + ndims*[1])
        ))
    unbiased_noise_psd = condition_covariance(unbiased_noise_psd, 1e-10)
    bf_vector = get_bf_vector(
        beamformer,
        unbiased_targed_psd.reshape(-1, shape[-2], shape[-2]),
        unbiased_noise_psd.reshape(-1, shape[-2], shape[-2]),
        **beamformer_kwargs
    )
    bf_vector = bf_vector.reshape(observation.shape[:-1])
    cleaned = apply_beamforming_vector(bf_vector, observation)
    if return_bf_vector:
        bf_vector = np.repeat(bf_vector[..., None], block_size, axis=-1)
        return np.swapaxes(
            morph('t...b->...t*b', cleaned)[..., :shape[-1]],
            axis1=-1, axis2=-2
        ), morph('t...b->...t*b', bf_vector)[..., :shape[-1]],
    else:
        return np.swapaxes(
            morph('t...b->...t*b', cleaned)[..., :shape[-1]],
            axis1=-1, axis2=-2
        )
