import numpy as np
import itertools
from nt.speech_enhancement.noise import get_energy, get_snr
from scipy.special import perm


__all__ = ['get_snr', 'input_sxr', 'output_sxr']


def _sxr(S, X):
    """ Calculate signal to `X` ratio

    :param S: Signal power
    :param X: X power
    :return: SXR
    """
    with np.errstate(divide='ignore'):
        result = 10 * np.log10(S/X)
    return result


def input_sxr(images, noise, average_sources=True, *, return_dict=False):
    """ Calculate input SXR values according to Tran Vu.

    The SXR definition is inspired by E. Vincent but the
    exact procedure as in his paper is not used. Dang Hai Tran Vu
    recommends this approach. Talk to Lukas Drude for details.

    Take the clean images and the noise at the sensors after applying a
    room impulse response and before mixing.

    :param images: Array of unmixed but reverberated speech in time domain
    :type images: np.ndarray with #Samples x #Sensors x #Speakers
    :param noise: Array of ground truth noise
    :type noise: np.ndarray with #Samples x #Sensors
    :param average_sources: Logical if SXR is average of speakers
    :type average_sources: bool
    :param return_dict: specifies if the returned value is a list or a dict.
                        If return_dict is a str, it is the prefix for the
                        dict keys (i.e. 'input_').
    :type return_dict: bool or str

    :return: SDR, SIR, SNR or {'sdr': SDR, 'sir': SIR, 'snr': SNR}
    """
    # TODO: This is not quite the correct way when utterances have different
    # len

    D = images.shape[1]  # Number of sensors
    K = images.shape[2]  # Number of speakers

    S = np.zeros((K, D))  # Signal power
    I = np.zeros((K, D))  # Interference power
    N = np.zeros(D)  # Noise power

    for d in range(D):
        for k in range(K):
            S[k, d] = get_energy(images[:, d, k], axis=0)
            I[k, d] = np.sum(
                get_energy(images[:, d, [n for n in range(K) if n != k]],
                           axis=0))
        N[d] = get_energy(noise[:, d], axis=0)

    if average_sources:
        S = np.mean(S)
        I = np.mean(I)
    else:
        S = np.mean(S, axis=1)
        I = np.mean(I, axis=1)

    N = np.mean(N)

    SDR = _sxr(S, I + N)
    SIR = _sxr(S, I)
    SNR = _sxr(S, N)

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
        return SDR, SIR, SNR


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
    :type image_contribution: #Samples x #sourceSpeakers x #targetSpeakers
    :param noise_contribution: Put the ground truth noise into the separation
      algorithm with fixed parameters. The output is `noise_contribution`.
    :type noise_contribution: #Samples times #targetSpeakers
    :param average_sources: Scalar logical if SXR is average of speakers;
      optional (default: true). If set to true, SXR-values are scalars.
    :param return_dict: specifies if the returned value is a list or a dict.
                        If return_dict is a str, it is the prefix for the
                        dict keys
    :type return_dict: bool or str

    :return SDR: #Sources times 1 vector of Signal to Distortion Ratios
    :return SIR: #Sources times 1 vector of Signal to Interference Ratios
    :return SNR: #Sources times 1 vector of Signal to Noise Ratios
   """
    # Assume, that the maximum number of speakers is smaller than 10.
    assert(image_contribution.shape[1] < 10)
    assert(image_contribution.shape[2] < 10)
    assert(noise_contribution.shape[1] < 10)

    K_source = image_contribution.shape[1]
    K_target = image_contribution.shape[2]

    S = np.zeros((K_source, K_target))
    N = np.zeros(K_target)

    for k_target in range(K_target):
        for k_source in range(K_source):
            S[k_source, k_target] = get_energy(
                image_contribution[:, k_source, k_target], axis=0
            )
        N[k_target] = get_energy(noise_contribution[:, k_target], axis=0)

    # We actually do not need to go through all permutations but through
    # all possible selections (picks) of the output sources to find that pick
    # which best matches the oracle sources.
    all_target_selections = np.array(
        list(itertools.permutations(range(K_target), r=K_source))
    )
    assert all_target_selections.shape == (perm(K_target, K_source), K_source), (
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
    for k_source in range(K_source):
        SS[k_source] = S[k_source, selection[k_source]]
        II[k_source] = np.sum(
            np.delete(S[:, selection[k_source]], k_source, axis=0)
        )
    NN = N[selection]

    if average_sources:
        SS = np.mean(SS)
        II = np.mean(II)
        NN = np.mean(NN)

    SDR = _sxr(SS, II + NN)
    SIR = _sxr(SS, II)
    SNR = _sxr(SS, NN)

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
        return SDR, SIR, SNR
