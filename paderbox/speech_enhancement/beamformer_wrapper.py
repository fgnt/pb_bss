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


def _get_rank_1_appoximation(atf_type, target_psd_matrix, noise_psd_matrix,
                             **atf_kwargs):
    if atf_type == 'rank1_pca':
        return get_pca_rank_one_estimate(target_psd_matrix, **atf_kwargs)
    elif atf_type == 'rank1_gev':
        return get_gev_rank_one_estimate(
            target_psd_matrix, noise_psd_matrix, **atf_kwargs)
    else:
        raise ValueError(atf_type, 'use either rank1_pca or rank1_gev')


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
        beamformer.replace('+ban', '')
    else:
        ban = False

    if beamformer == 'pca':
        bf_vec = get_pca_vector(target_psd_matrix, **bf_kwargs)
    elif beamformer in ['pca+mvdr', 'scaled_gev_atf+mvdr']:
        atf, _ = beamformer.split('+')
        atf_vector = _get_atf_vector(
            atf, target_psd_matrix,
            noise_psd_matrix,
            **bf_kwargs.pop('atf_kwargs', {})
        )
        bf_vec = get_mvdr_vector(atf_vector, noise_psd_matrix)
    elif beamformer in ['mvdr_souden', 'rank1_pca+mvdr_souden',
                        'rank1_gev+mvdr_souden']:
        if not beamformer == 'mvdr_souden':
            rank1_type, _ = beamformer.split('+')
            target_psd_matrix = _get_rank_1_appoximation(
                rank1_type,
                target_psd_matrix,
                noise_psd_matrix,
                **bf_kwargs.pop('atf_kwargs', {})
            )
        bf_vec = get_mvdr_vector_souden(target_psd_matrix, noise_psd_matrix,
                                        **bf_kwargs)
    elif beamformer in ['gev', 'rank1_pca+gev', 'rank1_gev+gev']:
        # rank1_gev+gev is not supported since it should no differ from gev
        if not beamformer == 'gev':
            rank1_type, _ = beamformer.split('+')
            target_psd_matrix = _get_rank_1_appoximation(
                rank1_type,
                target_psd_matrix,
                noise_psd_matrix,
                **bf_kwargs.pop('atf_kwargs', {})
            )
        bf_vec = get_gev_vector(target_psd_matrix, noise_psd_matrix,
                                **bf_kwargs)
    elif beamformer in ['wmwf', 'rank1_pca+wmwf', 'rank1_gev+wmwf']:
        if not beamformer == 'wmwf':
            rank1_type, _ = beamformer.split('+')
            target_psd_matrix = _get_rank_1_appoximation(
                rank1_type,
                target_psd_matrix,
                noise_psd_matrix,
                **bf_kwargs.pop('atf_kwargs', {})
            )
        bf_vec = get_wmwf_vector(target_psd_matrix, noise_psd_matrix,
                                 **bf_kwargs)
    else:
        raise ValueError('Unknown beamformer name', beamformer)

    if ban:
        bf_vec = blind_analytic_normalization(bf_vec, noise_psd_matrix)

    return bf_vec


def get_multi_source_bf_vector(
        beamformer,
        target_psd_matrix,
        noise_psd_matrix=None,
        **bf_kwargs
):
    if beamformer in ['pca+lcmv', 'scaled_gev_atf+lcmv']:
        assert 'response_vector' in bf_kwargs, bf_kwargs
        atf, _ = beamformer.split('+')
        atf_vector = _get_atf_vector(
            atf,
            target_psd_matrix,
            noise_psd_matrix,
            **bf_kwargs.pop('atf_kwargs', {})
        )
        bf_vec = get_lcmv_vector(atf_vector, noise_psd_matrix=noise_psd_matrix,
                                 **bf_kwargs)
    elif beamformer in ['lcmv_souden', 'rank1_pca+lcmv_souden',
                        'rank1_gev+lcmv_souden']:
        assert 'interference_psd_matrix' in bf_kwargs, bf_kwargs
        if not beamformer == 'lcmv_souden':
            rank1_type, _ = beamformer.split('+')
            target_psd_matrix = _get_rank_1_appoximation(
                rank1_type,
                target_psd_matrix,
                noise_psd_matrix,
                **bf_kwargs.pop('atf_kwargs', {})
            )
        bf_vec = get_lcmv_vector_souden(
            target_psd_matrix, noise_psd_matrix=noise_psd_matrix, **bf_kwargs)
    else:
        raise ValueError('Unknown beamformer name', beamformer)


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
