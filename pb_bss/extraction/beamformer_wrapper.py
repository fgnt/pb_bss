import numpy as np

from pb_bss.utils import labels_to_one_hot
from .beamformer import *

__all__ = [
    'get_bf_vector',
]


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
    assert noise_covariance_matrix is not None

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
    response_vector = labels_to_one_hot(
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
    """ Light wrapper to obtain a beamforming vector.

    Common beamformers:
     - 'mvdr_souden'
     - 'mvdr_souden+ban'
     - 'rank1_gev+mvdr_souden+ban'
     - 'gev_ban'

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
    elif 'ch' in beamformer_core and beamformer_core[2:].isdigit():
        D = target_psd_matrix.shape[-1]
        beamforming_vector = np.zeros(D)
        beamforming_vector[int(beamformer_core[2:])] = 1
        beamforming_vector = np.broadcast_to(
            beamforming_vector, target_psd_matrix.shape[:-1])
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
