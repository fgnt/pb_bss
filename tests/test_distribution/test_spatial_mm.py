import numpy as np

from einops import rearrange
from nara_wpe.utils import stft as _stft, istft as _istft

from pb_bss.testing.dummy_data import low_reverberation_data
from pb_bss.testing.dummy_data import reverberation_data
from pb_bss.evaluation.wrapper import InputMetrics, OutputMetrics
from pb_bss.permutation_alignment import DHTVPermutationAlignment
from pb_bss.distribution import (
    CACGMMTrainer,
    CBMMTrainer,
    CWMMTrainer,
)


def stft(signal):
    return _stft(signal, 512, 128)


def istft(signal, num_samples):
    return _istft(signal, 512, 128)[..., :num_samples]


def trainer_on_simulated_speech_data(
        Trainer=CACGMMTrainer,
        iterations=40,
        reverberation=False,
):
    reference_channel = 0
    sample_rate = 8000

    if reverberation:
        ex = reverberation_data()
    else:
        ex = low_reverberation_data()
    observation = ex['audio_data']['observation']
    Observation = stft(observation)
    num_samples = observation.shape[-1]

    Y_mm = rearrange(Observation, 'd t f -> f t d')

    t = Trainer()
    affiliation = t.fit(
        Y_mm,
        num_classes=3,
        iterations=iterations * 2,
        weight_constant_axis=-1,
    ).predict(Y_mm)
    
    pa = DHTVPermutationAlignment.from_stft_size(512)
    affiliation_pa = pa(rearrange(affiliation, 'f k t -> k f t'))
    affiliation_pa = rearrange(affiliation_pa, 'k f t -> k t f')

    Speech_image_0_est, Speech_image_1_est, Noise_image_est = Observation[reference_channel, :, :] * affiliation_pa

    speech_image_0_est = istft(Speech_image_0_est, num_samples=num_samples)
    speech_image_1_est = istft(Speech_image_1_est, num_samples=num_samples)
    noise_image_est = istft(Noise_image_est, num_samples=num_samples)

    ###########################################################################
    # Calculate the metrics

    speech_image = ex['audio_data']['speech_image']
    noise_image = ex['audio_data']['noise_image']
    speech_source = ex['audio_data']['speech_source']

    Speech_image = stft(speech_image)
    Noise_image = stft(noise_image)

    Speech_contribution = Speech_image[:, reference_channel, None, :, :] * affiliation_pa
    Noise_contribution = Noise_image[reference_channel, :, :] * affiliation_pa

    speech_contribution = istft(Speech_contribution, num_samples=num_samples)
    noise_contribution = istft(Noise_contribution, num_samples=num_samples)

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


def test_cacgmm():
    np.random.seed(0)
    scores = trainer_on_simulated_speech_data(CACGMMTrainer)
    np.testing.assert_allclose(
        scores['invasive_sxr_sdr'], [9.17896615, 17.02960108],
        err_msg=str(scores))
    np.testing.assert_allclose(
        scores['mir_eval_sxr_sdr'], [8.24826038, 12.53989719],
        err_msg=str(scores))

    np.random.seed(0)
    scores = trainer_on_simulated_speech_data(CACGMMTrainer, reverberation=True)
    np.testing.assert_allclose(
        scores['invasive_sxr_sdr'], [7.646699, 6.755594],
        err_msg=str(scores))
    np.testing.assert_allclose(
        scores['mir_eval_sxr_sdr'], [5.27172 , 5.915786],
        err_msg=str(scores))


def test_cwgmm():
    np.random.seed(0)
    scores = trainer_on_simulated_speech_data(CWMMTrainer)
    np.testing.assert_allclose(
        scores['invasive_sxr_sdr'], [17.47441, 20.946751],
        err_msg=str(scores))
    np.testing.assert_allclose(
        scores['mir_eval_sxr_sdr'], [9.675817, 13.557824],
        err_msg=str(scores))

    np.random.seed(0)
    scores = trainer_on_simulated_speech_data(CWMMTrainer, reverberation=True)
    np.testing.assert_allclose(
        scores['invasive_sxr_sdr'], [3.02768, 4.612752],
        err_msg=str(scores), rtol=1e-6)
    np.testing.assert_allclose(
        scores['mir_eval_sxr_sdr'], [2.50231548, 3.08808406],
        err_msg=str(scores))


def test_cbgmm():
    np.random.seed(0)
    # Bingham is very slow -> use only 2 iterations to test executable
    scores = trainer_on_simulated_speech_data(CBMMTrainer, iterations=2)
    np.testing.assert_allclose(
        scores['invasive_sxr_sdr'], [-0.51113, -3.246796],
        err_msg=str(scores))
    np.testing.assert_allclose(
        scores['mir_eval_sxr_sdr'], [9.675817, 13.557824],
        err_msg=str(scores))
