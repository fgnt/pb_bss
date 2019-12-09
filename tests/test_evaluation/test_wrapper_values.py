import numpy as np
import scipy.signal

from pb_bss.evaluation.wrapper import InputMetrics, OutputMetrics


def scenario():
    samples = 10_000
    rir_length = 4
    channels = 3
    speakers = 2
    np.random.seed(1)

    speech_source_1 = np.random.rand(samples)
    speech_source_2 = np.random.rand(samples)

    h1 = np.random.rand(channels, rir_length)
    h2 = np.random.rand(channels, rir_length)

    speech_image_1 = np.array([
        scipy.signal.fftconvolve(speech_source_1, h, mode='same')
        for h in h1
    ])
    speech_image_2 = np.array([
        scipy.signal.fftconvolve(speech_source_2, h, mode='same')
        for h in h2
    ])

    assert speech_image_2.shape == (channels, samples)

    noise = 0.01 * np.random.rand(channels, samples)

    observation = speech_image_1 + speech_image_2 + noise

    return {
        'speech_source': np.array([speech_source_1, speech_source_2]),
        'speech_image': np.array([speech_image_1, speech_image_2]),
        'noise_image': noise,
        'observation': observation
    }


def test_input_metrics():
    example = scenario()
    metrics = InputMetrics(
            observation=example['observation'],
            speech_source=example['speech_source'],
            speech_image=example['speech_image'],
            noise_image=example['noise_image'],
            sample_rate=8000,

    )

    assert metrics.K_source == 2
    assert metrics.channels == 3

    for k, v in metrics.as_dict().items():
        if k == 'invasive_sxr_sdr':
            np.testing.assert_allclose(
                v, [[ 4.634096,  1.821645,  5.012743],
                    [-4.634303, -1.821825, -5.013139]], rtol=1e-6)
        elif k == 'invasive_sxr_sir':
            np.testing.assert_allclose(
                v, [[ 4.63425 ,  1.821754,  5.013044],
                    [-4.63425 , -1.821754, -5.013044]], rtol=1e-6)
        elif k == 'invasive_sxr_snr':
            np.testing.assert_allclose(
                v, [[49.137625, 47.859369, 46.598417],
                    [44.503376, 46.037615, 41.585373]])
        elif k == 'mir_eval_sxr_sdr':
            np.testing.assert_allclose(
                v, [[16.286314, 15.048399, 17.420134],
                    [14.386505, 14.606471, 12.842921]])
        elif k == 'mir_eval_sxr_sir':
            np.testing.assert_allclose(
                v, [[18.172265, 17.323722, 18.868235],
                    [15.523357, 16.609909, 13.310729]])
        elif k == 'mir_eval_sxr_sar':
            np.testing.assert_allclose(
                v, [[20.883413, 19.02361 , 22.949934],
                    [20.883413, 19.02361 , 22.949934]])
        elif k == 'pesq':
            np.testing.assert_allclose(
                v, [[3.494761, 3.034838, 3.755455],
                    [2.437896, 2.820094, 2.434496]], rtol=1e-6)
        elif k == 'stoi':
            np.testing.assert_allclose(
                v, [[0.691546, 0.626544, 0.717809],
                    [0.28424 , 0.345368, 0.279996]], rtol=1e-5)
        else:
            raise KeyError(k, v)


def test_output_metrics():
    example = scenario()

    # Take speech image + noise as prediction, i.e. perfect croos talber suppression
    speech_prediction = example['speech_image'][..., 0, :] + example['noise_image'][..., 0, :]

    speech_image_1, speech_image_2 = example['speech_image'][..., 0, :]

    speech_contribution = np.array([
        [speech_image_1, np.zeros_like(speech_image_2)],
        [np.zeros_like(speech_image_1), speech_image_2],
    ])
    noise_contribution = np.array([
        example['noise_image'][..., 0, :],
        example['noise_image'][..., 0, :],
    ])

    metrics = OutputMetrics(
            speech_prediction=speech_prediction,
            # observation=example['observation'],
            speech_source=example['speech_source'],
            # speech_image=example['speech_image'],
            # noise_image=example['noise_image'],
            speech_contribution=speech_contribution,
            noise_contribution=noise_contribution,
            sample_rate=8000,
            # channel_score_reduce='mean',
    )

    assert metrics.K_source == 2

    for k, v in metrics.as_dict().items():
        if k == 'invasive_sxr_sdr':
            np.testing.assert_allclose(v, [49.137625, 44.503376])
        elif k == 'invasive_sxr_sir':
            np.testing.assert_allclose(v, np.inf)
        elif k == 'invasive_sxr_snr':
            np.testing.assert_allclose(v, [49.137625, 44.503376])
        elif k == 'mir_eval_sxr_sdr':
            np.testing.assert_allclose(v, [17.071665, 24.711722])
        elif k == 'mir_eval_sxr_sir':
            np.testing.assert_allclose(v, [29.423133, 37.060289])
        elif k == 'mir_eval_sxr_sar':
            np.testing.assert_allclose(v, [17.336992, 24.973125])
        elif k == 'pesq':
            np.testing.assert_allclose(v, [4.37408 , 4.405752])
        elif k == 'stoi':
            np.testing.assert_allclose(v, [0.968833, 0.976151], rtol=1e-6)
        elif k == 'mir_eval_sxr_selection':
            assert all(v == [0, 1])
        else:
            raise KeyError(k, v)
