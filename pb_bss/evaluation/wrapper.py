import cached_property

import numpy as np

from einops import rearrange


def _get_err_msg(msg, metrics: 'Metrics'):
    msg = f'{msg}'
    msg += f'\nShapes: (is shape) (symbolic shape)'
    msg += f'\n\tspeech_prediction: {metrics.speech_prediction.shape} (N)'
    msg += f'\n\tspeech_source: {metrics.speech_source.shape} (K_source, K_target, N)'
    msg += f'\n\tspeech_contribution: {metrics.speech_contribution.shape} (K_target, N)'
    msg += f'\n\tnoise_contribution: {metrics.noise_contribution.shape} (K_source, N)'
    return msg


class Metrics:
    def __init__(
            self,
            speech_prediction: 'Shape(K_target, N)',
            speech_source: 'Shape(K_source, N)',
            speech_contribution: 'Shape(K_source, K_target, N)'=None,
            noise_contribution: 'Shape(K_target, N)'=None,
            sample_rate: int = None,
    ):
        assert speech_prediction.ndim == 2, speech_prediction.shape
        assert speech_source.ndim == 2, speech_source.shape

        self.speech_prediction = speech_prediction
        self.speech_source = speech_source
        self.speech_contribution = speech_contribution
        self.noise_contribution = noise_contribution
        self.sample_rate = sample_rate

        samples = speech_prediction.shape[-1]

        # if speech_contribution is not None:
        #     K_source, K_target, samples_ = speech_contribution.shape
        #     assert samples == samples_, _get_err_msg()
        #     ktaget_, samples_ = enhanced_noise_image.shape
        #     assert samples == samples_, get_msg((samples, samples_))
        #     assert ktaget == ktaget_, get_msg((ktaget, ktaget_))
        #
        #     assert ksource < 5, get_msg(ksource)
        #     assert ktaget < 5, get_msg(ktaget)
        #
        #     ksource_, samples_ = speech_source.shape
        #     assert samples == samples_, get_msg((samples, samples_))
        #     assert ksource == ksource_, get_msg((ksource, ksource_))


    @cached_property.cached_property
    def enhanced_speech(self):
        assert self.speech_prediction.ndim == 2, self.speech_prediction.shape
        assert self.speech_prediction.shape[0] < 10, self.speech_prediction.shape
        assert self.speech_prediction.shape[0] == len(self.selection) + 1, self.speech_prediction.shape
        return self.speech_prediction[self.selection]

    @cached_property.cached_property
    def mir_eval(self):
        import paderbox as pb
        return pb.evaluation.mir_eval_sources(
            reference=self.speech_source,
            estimation=self.speech_prediction,
            return_dict=True,
        )

    @cached_property.cached_property
    def pesq_nb(self):
        import paderbox as pb
        try:
            return pb.evaluation.pesq(
                reference=self.speech_source,
                degraded=self.enhanced_speech,
                rate=self.sample_rate,
                mode='nb',
            )
        except OSError:
            return np.nan

    @cached_property.cached_property
    def pesq_wb(self):
        import paderbox as pb
        try:
            return pb.evaluation.pesq(
                reference=self.speech_source,
                degraded=self.enhanced_speech,
                rate=self.sample_rate,
                mode='nb',
            )
        except OSError:
            return np.nan

    @cached_property.cached_property
    def pypesq_nb(self):
        try:
            import pypesq
        except ImportError:
            raise AssertionError(
                'To use this pesq, install '
                'https://github.com/ludlows/python-pesq .'
            )

        assert self.speech_source.shape == self.enhanced_speech.shape, (self.speech_source.shape, self.enhanced_speech.shape)
        assert self.speech_source.ndim == 2, (self.speech_source.shape, self.enhanced_speech.shape)
        assert self.speech_source.shape[0] < 5, (self.speech_source.shape, self.enhanced_speech.shape)

        return [
            pypesq.pypesq(ref=ref, deg=deg, fs=self.sample_rate, mode='nb')
            for ref, deg in zip(self.speech_source, self.enhanced_speech)
        ]

    @cached_property.cached_property
    def pypesq_wb(self):
        try:
            import pypesq
        except ImportError:
            raise AssertionError(
                'To use this pesq, install '
                'https://github.com/ludlows/python-pesq .'
            )

        assert self.speech_source.shape == self.enhanced_speech.shape, (self.speech_source.shape, self.enhanced_speech.shape)
        assert self.speech_source.ndim == 2, (self.speech_source.shape, self.enhanced_speech.shape)
        assert self.speech_source.shape[0] < 5, (self.speech_source.shape, self.enhanced_speech.shape)

        return [
            pypesq.pypesq(ref=ref, deg=deg, fs=self.sample_rate, mode='wb')
            for ref, deg in zip(self.speech_source, self.enhanced_speech)
        ]

    @cached_property.cached_property
    def selection(self):
        return self.mir_eval['permutation']

    @cached_property.cached_property
    def sxr(self):
        import paderbox as pb
        invasive_sxr = pb.evaluation.output_sxr(
            # rearrange(beamformed_clean, 'ksource ktaget samples -> ktaget ksource samples'),
            rearrange(
                self.speech_contribution,
                'ksource ktaget samples -> ksource ktaget samples'
            )[:, self.selection, :],
            rearrange(
                self.noise_contribution, 'ktaget samples -> ktaget samples'
            )[self.selection, :],
            return_dict=True,
        )
        return invasive_sxr
