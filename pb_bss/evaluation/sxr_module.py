import numpy as np
import collections
import itertools
from paderbox.speech_enhancement.noise import (
    get_variance_for_zero_mean_signal,
    get_snr,
)
from scipy.special import perm


__all__ = ['get_snr', 'input_sxr', 'output_sxr']


ResultTuple = collections.namedtuple('SXR', ['sdr', 'sir', 'snr'])


def _sxr(S, X):
    """ Calculate signal to `X` ratio

    :param S: Signal power
    :param X: X power
    :return: SXR
    """
    with np.errstate(divide='ignore'):
        result = 10 * np.log10(S / X)
    return result


def input_sxr(
        images,
        noise,
        average_sources=True,
        average_channels=True,
        *,
        return_dict=False
):
    """ Calculate input SXR values according to Tran Vu.

    The SXR definition is inspired by E. Vincent but the
    exact procedure as in his paper is not used. Dang Hai Tran Vu
    recommends this approach. Talk to Lukas Drude for details.

    Take the clean images and the noise at the sensors after applying a
    room impulse response and before mixing.

    :param images: Array of unmixed but reverberated speech in time domain
    :type images: np.ndarray with #speakers x #sensors x #samples
    :param noise: Array of ground truth noise
    :type noise: np.ndarray with #sensors x #samples
    :param average_sources: Logical if SXR is average of speakers
    :type average_sources: bool
    :param return_dict: specifies if the returned value is a list or a dict.
                        If return_dict is a str, it is the prefix for the
                        dict keys (i.e. 'input_').
    :type return_dict: bool or str

    :return: SDR, SIR, SNR or {'sdr': SDR, 'sir': SIR, 'snr': SNR}
    """
    # TODO: Not really the correct way when utterances have different length.

    K, D, T = images.shape  # Number of speakers, sensors, samples

    assert (D, T) == noise.shape, (images.shape, noise.shape)
    assert K < 10, images.shape
    assert D < 30, images.shape

    S = get_variance_for_zero_mean_signal(images, axis=-1)  # Signal power
    I = np.zeros((K, D))  # Interference power
    N = get_variance_for_zero_mean_signal(noise, axis=-1)  # Noise power

    for d in range(D):
        for k in range(K):
            I[k, d] = np.sum(
                S[[n for n in range(K) if n != k], d],
                axis=0
            )

    if average_channels:
        S, I, N = [np.mean(power, axis=-1) for power in (S, I, N)]

    SDR = _sxr(S, I + N)
    SIR = _sxr(S, I)
    SNR = _sxr(S, N)

    if average_sources:
        SDR = np.mean(SDR, axis=0)
        SIR = np.mean(SIR, axis=0)
        SNR = np.mean(SNR, axis=0)
        
    if return_dict:
        if return_dict is True:
            return {'sdr': SDR, 'sir': SIR, 'snr': SNR}
        elif isinstance(return_dict, str):
            return {return_dict + 'sdr': SDR,
                    return_dict + 'sir': SIR,
                    return_dict + 'snr': SNR}
        else:
            raise TypeError(return_dict)
    else:
        return ResultTuple(SDR, SIR, SNR)


def output_sxr(image_contribution, noise_contribution, average_sources=True,
               return_dict=False):
    """ Calculate output SXR values.

    The SXR definition is inspired by E. Vincent but the
    exact procedure as in his paper is not used. Dang Hai Tran Vu
    recommends this approach. Talk to Lukas Drude for details.

    The output signal of the system under test can be decomposed in
    contributions due to the speakers and noise. The two input signals
    are the processed images and noise by the complete separation
    system.

    Use the mixed signals to run your algorithm. Save the estimated separation
    parameters (i.e. beamforming vectors, gain functions, ...) and apply the
    algorithm with the fixed parameters to the clean images and ground truth
    noise separately. Evaluate the results with this function to obtain
    intrusive SXR measures.

    :param image_contribution: Put each of the clean images into the
      separation algorithm with fixed parameters. The output of the separation
      algorithm can now be used as `image_contribution`.
    :type image_contribution: #source_speakers x #target_speakers x #samples
    :param noise_contribution: Put the ground truth noise into the separation
      algorithm with fixed parameters. The output is `noise_contribution`.
    :type noise_contribution: #target_speakers x #samples
    :param average_sources: Scalar logical if SXR is average of speakers;
      optional (default: true). If set to true, SXR-values are scalars.
    :param return_dict: specifies if the returned value is a list or a dict.
                        If return_dict is a str, it is the prefix for the
                        dict keys
    :type return_dict: bool or str

    :return SDR: #source_speakers vector of Signal to Distortion Ratios
    :return SIR: #source_speakers vector of Signal to Interference Ratios
    :return SNR: #source_speakers vector of Signal to Noise Ratios

    """

    K_source, K_target, samples = image_contribution.shape

    # Chech that image_contribution.shape and noise_contribution.shape match
    assert noise_contribution.shape == (K_target, samples), (
        image_contribution.shape, noise_contribution.shape
    )

    # Assume, that the maximum number of speakers is smaller than 10.
    assert K_source < 10, (image_contribution.shape, noise_contribution.shape)
    assert K_target < 10, (image_contribution.shape, noise_contribution.shape)

    S = get_variance_for_zero_mean_signal(image_contribution, axis=-1)
    N = get_variance_for_zero_mean_signal(noise_contribution, axis=-1)

    # We actually do not need to go through all permutations but through
    # all possible selections (picks) of the output sources to find that pick
    # which best matches the oracle sources.
    all_target_selections = np.array(
        list(itertools.permutations(range(K_target), r=K_source))
    )
    assert all_target_selections.shape == (
        perm(K_target, K_source), K_source
    ), (
        all_target_selections.shape, perm(K_target, K_source), K_source
    )

    mutual_power = np.zeros(all_target_selections.shape[0])

    for p in range(all_target_selections.shape[0]):
        mutual_power[p] = np.sum([
            S[k_source, all_target_selections[p, k_source]]
            for k_source in range(K_source)
        ])

    max_idx = np.argmax(mutual_power)
    selection = all_target_selections[max_idx]

    SS = np.zeros(K_source)
    II = np.zeros(K_source)

    # CB: Use advanced indexing to remove loop?
    for k_source in range(K_source):
        SS[k_source] = S[k_source, selection[k_source]]
        II[k_source] = np.sum(
            np.delete(S[:, selection[k_source]], k_source, axis=0)
        )
    NN = N[selection]

    SDR = _sxr(SS, II + NN)
    SIR = _sxr(SS, II)
    SNR = _sxr(SS, NN)

    if average_sources:
        SDR = np.mean(SDR)
        SIR = np.mean(SIR)
        SNR = np.mean(SNR)

    if return_dict is True:
        if return_dict is True:
            return {'sdr': SDR, 'sir': SIR, 'snr': SNR}
        elif isinstance(return_dict, str):
            return {return_dict + 'sdr': SDR,
                    return_dict + 'sir': SIR,
                    return_dict + 'snr': SNR}
        else:
            raise TypeError(return_dict)
    else:
        return ResultTuple(SDR, SIR, SNR)
