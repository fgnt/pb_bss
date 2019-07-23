import cached_property

import numpy as np

from einops import rearrange
import pb_bss


def _get_err_msg(msg, metrics: 'Metrics'):
    msg = f'{msg}'
    msg += f'\nShapes: (is shape) (symbolic shape)'
    msg += f'\n\tspeech_prediction: {metrics.speech_prediction.shape} (N)'
    msg += f'\n\tspeech_source: {metrics.speech_source.shape} (K_source, N)'
    if metrics.speech_contribution is not None:
        msg += f'\n\tspeech_contribution: {metrics.speech_contribution.shape} (K_source, K_target, N)'
    if metrics.noise_contribution is not None:
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

        # The remaining init are only asserts to check the shapes

        samples = self.speech_prediction.shape[-1]
        K_source = self.speech_source.shape[0]
        K_target = self.speech_prediction.shape[0]

        assert K_source <= 5, _get_err_msg(
            f'Number of source speakers (K_source) of speech_source is '
            f'{K_source}. Expect a reasonable value of 5 or less.'
        )
        assert K_target <= 5, _get_err_msg(
            f'Number of target speakers (K_target) of speech_prediction is '
            f'{K_target}. Expect a reasonable value of 5 or less.'
        )
        assert K_source in [K_target, K_target+1], _get_err_msg(
            f'Number of source speakers (K_source) should be equal to'
            f'number of target speakers (K_target) or K_target + 1'
        )
        assert self.speech_source.shape[0] == samples, _get_err_msg(
            'Num samples (N) of speech_source do not fit to the'
            'shape from speech_prediction'
        )
        if speech_contribution is not None and noise_contribution is not None:
            assert noise_contribution is not None, noise_contribution

            K_source_, K_target_, samples_ = speech_contribution.shape
            assert samples == samples_, _get_err_msg(
                'Num samples (N) of speech_contribution do not fit to the'
                'shape from speech_prediction'
            )
            assert K_target == K_target_, _get_err_msg(
                'Num target speakers (K_target) of speech_contribution do not '
                'fit to the shape from speech_prediction'
            )
            assert K_source < 5, _get_err_msg(
                'Num source speakers (K_source) of speech_contribution do not '
                'fit to the shape from speech_source'
            )
            K_target_, samples_ = noise_contribution.shape
            assert samples == samples_, _get_err_msg(
                'Num samples (N) of noise_contribution do not fit to the'
                'shape from speech_prediction'
            )
            assert K_target == K_target_, _get_err_msg(
                'Num target speakers (K_target) of noise_contribution do not '
                'fit to the shape from speech_prediction'
            )
        else:
            assert speech_contribution is None and noise_contribution is None, (
                'Expect that speech_contribution and noise_contribution are '
                'both None or given.\n'
                'Got\n'
                f'speech_contribution: {speech_contribution}\n'
                f'noise_contribution: {noise_contribution}\n'
            )


    @cached_property.cached_property
    def selection(self):
        return self.mir_eval['permutation']

    @cached_property.cached_property
    def speech_prediction_selection(self):
        assert self.speech_prediction.ndim == 2, self.speech_prediction.shape
        assert self.speech_prediction.shape[0] < 10, self.speech_prediction.shape
        assert self.speech_prediction.shape[0] == len(self.selection) + 1, self.speech_prediction.shape
        return self.speech_prediction[self.selection]

    @cached_property.cached_property
    def mir_eval(self):
        return pb_bss.evaluation.mir_eval_sources(
            reference=self.speech_source,
            estimation=self.speech_prediction,
            return_dict=True,
        )

    @cached_property.cached_property
    def pesq(self):
        import paderbox as pb
        mode = {8000: 'nb', 16000: 'wb'}[self.sample_rate]
        try:
            return pb.evaluation.pesq(
                reference=self.speech_source,
                degraded=self.speech_prediction_selection,
                rate=self.sample_rate,
                mode=mode,
            )
        except OSError:
            return np.nan

    @cached_property.cached_property
    def pypesq(self):
        try:
            import pypesq
        except ImportError:
            raise AssertionError(
                'To use this pesq, install '
                'https://github.com/ludlows/python-pesq .'
            )
        mode = {8000: 'nb', 16000: 'wb'}[self.sample_rate]

        assert self.speech_source.shape == self.speech_prediction_selection.shape, (self.speech_source.shape, self.speech_prediction_selection.shape)
        assert self.speech_source.ndim == 2, (self.speech_source.shape, self.speech_prediction_selection.shape)
        assert self.speech_source.shape[0] < 5, (self.speech_source.shape, self.speech_prediction_selection.shape)

        return [
            pypesq.pypesq(ref=ref, deg=deg, fs=self.sample_rate, mode=mode)
            for ref, deg in zip(
                self.speech_source, self.speech_prediction_selection)
        ]

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

    @cached_property.cached_property
    def stoi(self):
        import paderbox as pb

        stoi = list()
        K = self.enhanced_speech.shape[0]
        for k in range(K):
            stoi.append(pb.evaluation.stoi(
                self.speech_source[k, :],
                self.enhanced_speech[k, :],
                sample_rate=self.sample_rate,
            ))
        return stoi

    def as_dict(self):
        return dict(
            mir_eval_sxr_sdr=self.mir_eval['sdr'],
            mir_eval_sxr_sir=self.mir_eval['sir'],
            mir_eval_sxr_sar=self.mir_eval['sar'],
            mir_eval_sxr_selection=self.mir_eval['selection'],
            pesq=self.pesq_nb,
            invasive_sxr_sdr=self.sxr['sdr'],
            invasive_sxr_sir=self.sxr['sir'],
            invasive_sxr_snr=self.sxr['snr'],
            stoi=self.stoi,
        )
