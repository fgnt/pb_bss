import numpy as np

from einops import rearrange
from nara_wpe.utils import stft as _stft, istft as _istft

from pb_bss.testing.dummy_data import low_reverberation_data
from pb_bss.evaluation.wrapper import InputMetrics, OutputMetrics
from pb_bss.permutation_alignment import DHTVPermutationAlignment
from pb_bss.distribution import (
    CACGMMTrainer,
    CBMMTrainer,
    CWMMTrainer,
)

import paderbox as pb


def stft(signal):
    return _stft(signal, 512, 128)


def istft(signal):
    return _istft(signal, 512, 128)


def test_trainer_on_simulated_speech_data(Trainer, iterations=40):
    """
    >>> test_asd(CACGMMTrainer)
    {'invasive_sxr_sdr': array([ 9.54582547, 13.5462911 ]), 'mir_eval_sxr_sdr': array([ 7.93209258, 11.43185283])}
    >>> test_asd(CBMMTrainer, iterations=2)  # Bingham is very slow
    {'invasive_sxr_sdr': array([0.06166788, 0.94633618]), 'mir_eval_sxr_sdr': array([ 0.0435892 , -2.44499831])}
    >>> test_asd(CWMMTrainer)
    {'invasive_sxr_sdr': array([17.50553326, 20.9246735 ]), 'mir_eval_sxr_sdr': array([ 9.70250815, 13.55070934])}

    """

    reference_channel = 0
    sample_rate = 8000

    ex = low_reverberation_data()
    observation = ex['audio_data']['observation']
    speech_image = ex['audio_data']['speech_image']
    noise_image = ex['audio_data']['noise_image']
    speech_source = ex['audio_data']['speech_source']

    Observation = stft(observation)
    Speech_image = stft(speech_image)

    # ex['audio_data']['noise_image'] = ex['audio_data']['observation'] - np.sum(ex['audio_data']['speech_image'], axis=0)
    Noise_image = stft(noise_image)

    ideal_masks = pb.speech_enhancement.ideal_ratio_mask(
        np.sqrt(np.abs(
            (np.array([*Speech_image, Noise_image]) ** 2)
        ).sum(1)),
        source_axis=0,
    )

    Y_mm = rearrange(Observation, 'd t f -> f t d')

    t = Trainer()
    affiliation = t.fit(
        Y_mm,
        num_classes=3,
        iterations=iterations * 2,
        weight_constant_axis=-1,
        #     initialization=rearrange(ex['audio_data']['ideal_masks'], 'k t f -> f k t'),
    ).predict(Y_mm)
    
    pa = DHTVPermutationAlignment.from_stft_size(512)
    affiliation_pa = pa(rearrange(affiliation, 'f k t -> k f t'))

    Speech_image_0_est = Observation[reference_channel, :, :].T * affiliation_pa[0, :, :]
    Speech_image_1_est = Observation[reference_channel, :, :].T * affiliation_pa[1, :, :]
    Noise_image_est = Observation[reference_channel, :, :].T * affiliation_pa[2, :, :]

    speech_image_0_est = istft(Speech_image_0_est.T)[..., :observation.shape[-1]]
    speech_image_1_est = istft(Speech_image_1_est.T)[..., :observation.shape[-1]]
    noise_image_est = istft(Noise_image_est.T)[..., :observation.shape[-1]]

    Speech_contribution = Speech_image[:, reference_channel, None, :, :] * rearrange(affiliation_pa, 'k f t -> k t f')
    Noise_contribution = Noise_image[reference_channel, :, :] * rearrange(affiliation_pa, 'k f t -> k t f')

    speech_contribution = istft(Speech_contribution)[..., :observation.shape[-1]]
    noise_contribution = istft(Noise_contribution)[..., :observation.shape[-1]]

    input_metric = InputMetrics(
        observation=observation,
        speech_source=speech_source,
        speech_image=speech_image,
        noise_image=noise_image,
        sample_rate=sample_rate,
    )

    output_metric = OutputMetrics(
        speech_prediction=np.array(
            [speech_image_0_est, speech_image_1_est, noise_image_est]),
        speech_source=speech_source,
        speech_contribution=speech_contribution,
        noise_contribution=noise_contribution,
        sample_rate=sample_rate,
    )

    return {
        'invasive_sxr_sdr': output_metric.invasive_sxr['sdr'] - input_metric.invasive_sxr['sdr'][:, reference_channel],
        'mir_eval_sxr_sdr': output_metric.mir_eval['sdr'] - input_metric.mir_eval['sdr'][:, reference_channel],
    }
