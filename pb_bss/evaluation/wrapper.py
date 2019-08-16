import cached_property

import numpy as np

from einops import rearrange
import pb_bss


# TODO: Should mir_eval_sxr_selection stay in InputMetrics?
# TODO: Add SI-SDR even though there are arguments against it.
# TODO: Explain, why we compare BSS-Eval against source and not image.
# TODO: Explain, why invasive SXR does not work with, e.g., Nara-WPE.


def _get_err_msg(msg, metrics: 'OutputMetrics'):
    msg = f'{msg}'
    msg += f'\nShapes: (is shape) (symbolic shape)'
    msg += f'\n\tspeech_prediction: {metrics.speech_prediction.shape} (K_target, N)'  # noqa
    msg += f'\n\tspeech_source: {metrics.speech_source.shape} (K_source, N)'
    if metrics.speech_contribution is not None:
        msg += (f'\n\tspeech_contribution: '
                f'{metrics.speech_contribution.shape} (K_source, K_target, N)')
    if metrics.noise_contribution is not None:
        msg += (f'\n\tnoise_contribution: '
                f'{metrics.noise_contribution.shape} (K_target, N)')
    return msg


class InputMetrics:
    def __init__(
            self,
            observation: 'Shape(D, N)',
            speech_source: 'Shape(K_source, N)',
            speech_image: 'Shape(K_source, D, N)'=None,
            noise_image: 'Shape(D, N)'=None,
            sample_rate: int = None,
            enable_si_sdr: bool = False,
    ):
        """

        Args:
            observation: When you pass D channels, you get D metrics per
                speaker. If you want to select a reference channel, you need
                to slice the input to just have a singleton channel dimension.
            speech_source:
            speech_image:
            noise_image:
            sample_rate:
            enable_si_sdr: Since SI-SDR is only well defined for non-reverb
                single-channel data, it is disabled by default.
        """
        self.observation = observation
        self.speech_source = speech_source
        self.speech_image = speech_image
        self.noise_image = noise_image
        self.sample_rate = sample_rate

        self._has_image_signals \
            = (speech_image is not None and noise_image is not None)

        self.samples = self.observation.shape[-1]
        self.channels = self.observation.shape[-2]
        self.K_source = self.speech_source.shape[0]

        self.enable_si_sdr = enable_si_sdr

        self.check_inputs()

    def check_inputs(self):
        assert self.observation.ndim == 2, self.observation.shape
        assert self.speech_source.ndim == 2, self.speech_source.shape

    @cached_property.cached_property
    def mir_eval(self):
        return pb_bss.evaluation.mir_eval_sources(
            reference=rearrange(
                [self.speech_source] * self.channels,
                'channels sources samples -> sources channels samples'
            ),
            estimation=rearrange(
                [self.observation] * self.K_source,
                'sources channels samples -> sources channels samples'
            ),
            return_dict=True,
            compute_permutation=False,
        )

    @cached_property.cached_property
    def pesq(self):
        try:
            import paderbox as pb
        except ImportError:
            return np.nan

        mode = {8000: 'nb', 16000: 'wb'}[self.sample_rate]
        try:
            scores = pb.evaluation.pesq(
                reference=rearrange(
                    [self.speech_source] * self.channels,
                    'channels sources samples -> (sources channels) samples'
                ),
                degraded=rearrange(
                    [self.observation] * self.K_source,
                    'sources channels samples -> (sources channels) samples'
                ),
                rate=self.sample_rate,
                mode=mode,
            )
            return np.reshape(scores, [self.K_source, self.channels])
        except OSError:
            return np.nan

    @cached_property.cached_property
    def invasive_sxr(self):
        from pb_bss.evaluation.sxr_module import input_sxr
        invasive_sxr = input_sxr(
            rearrange(
                self.speech_image,
                'sources sensors samples -> sources sensors samples'
            ),
            rearrange(self.noise_image, 'sensors samples -> sensors samples'),
            average_sources=False,
            average_channels=False,
            return_dict=True,
        )
        return invasive_sxr

    @cached_property.cached_property
    def stoi(self):
        scores = pb_bss.evaluation.stoi(
            reference=rearrange(
                [self.speech_source] * self.channels,
                'channels sources samples -> sources channels samples'
            ),
            estimation=rearrange(
                [self.observation] * self.K_source,
                'sources channels samples -> sources channels samples'
            ),
            sample_rate=self.sample_rate,
        )
        return scores

    @cached_property.cached_property
    def si_sdr(self):
        if self.enable_si_sdr:
            return pb_bss.evaluation.si_sdr(
                # Shape: (sources, 1, samples)
                reference=self.speech_source[:, None, :],
                # Shape: (1, channels, samples)
                estimation=self.observation[None, :, :],
            )
        else:
            raise ValueError(
                'SI-SDR is disabled by default since it is only well-defined '
                'for non-reverberant single-channel data. Enable it with '
                '`enable_si_sdr=True`.'
            )

    def as_dict(self):
        metrics = dict(
            pesq=self.pesq,
            stoi=self.stoi,
            mir_eval_sxr_sdr=self.mir_eval['sdr'],
            mir_eval_sxr_sir=self.mir_eval['sir'],
            mir_eval_sxr_sar=self.mir_eval['sar'],
        )

        if self.enable_si_sdr:
            metrics['si_sdr'] = self.si_sdr

        if self._has_image_signals:
            metrics['invasive_sxr_sdr'] = self.invasive_sxr['sdr']
            metrics['invasive_sxr_sir'] = self.invasive_sxr['sir']
            metrics['invasive_sxr_snr'] = self.invasive_sxr['snr']

        return metrics


class OutputMetrics:
    def __init__(
            self,
            speech_prediction: 'Shape(K_target, N)',
            speech_source: 'Shape(K_source, N)',
            speech_contribution: 'Shape(K_source, K_target, N)'=None,
            noise_contribution: 'Shape(K_target, N)'=None,
            sample_rate: int = None,
            enable_si_sdr: bool = False,
    ):
        """

        Args:
            speech_prediction: Shape(K_target, N)
                The prediction of the source signal.
            speech_source: Shape(K_source, N)
                The true source signal, before the reverberation.
            speech_contribution: Shape(K_source, K_target, N)
                Optional for linear enhancements. See below.
            noise_contribution: Shape(K_target, N)
                Optional for linear enhancements. See below.
            sample_rate: int
                pesq and stoi need the sample rate.
                In pesq the sample rate defines the mode:
                    8000: narrow band (nb)
                    8000: wide band (wb)
            enable_si_sdr: Since SI-SDR is only well defined for non-reverb
                single-channel data, it is disabled by default.

        speech_contribution and noise_contribution can only be calculated for
        linear system and are used for the calculation of invasive_sxr.
        Use speech image (reverberated speech source) and apply for each source
        the enhancement for each target speaker enhancement. The same for the
        noise and each target speaker.

        """
        self.speech_prediction = speech_prediction
        self.speech_source = speech_source
        self.speech_contribution = speech_contribution
        self.noise_contribution = noise_contribution
        self.sample_rate = sample_rate

        self._has_contribution_signals = (
            speech_contribution is not None
            and
            noise_contribution is not None
        )

        self.samples = self.speech_prediction.shape[-1]
        self.K_source = self.speech_source.shape[0]
        self.K_target = self.speech_prediction.shape[0]

        self.enable_si_sdr = enable_si_sdr

        self.check_inputs()

    def check_inputs(self):
        assert self.speech_prediction.ndim == 2, self.speech_prediction.shape
        assert self.speech_source.ndim == 2, self.speech_source.shape

        assert self.K_source <= 5, _get_err_msg(
            f'Number of source speakers (K_source) of speech_source is '
            f'{self.K_source}. Expect a reasonable value of 5 or less.',
            self
        )
        assert self.K_target <= 5, _get_err_msg(
            f'Number of target speakers (K_target) of speech_prediction is '
            f'{self.K_target}. Expect a reasonable value of 5 or less.',
            self
        )
        assert self.K_target in [self.K_source, self.K_source+1], _get_err_msg(
            f'Number of target speakers (K_target) should be equal to '
            f'number of source speakers (K_source) or K_target + 1',
            self
        )
        assert self.speech_source.shape[1] == self.samples, _get_err_msg(
            'Num samples (N) of speech_source does not fit to the'
            'shape from speech_prediction',
            self
        )
        if (
            self.speech_contribution is not None
            and self.noise_contribution is not None
        ):
            assert self.noise_contribution is not None, self.noise_contribution

            K_source_, K_target_, samples_ = self.speech_contribution.shape
            assert self.samples == samples_, _get_err_msg(
                'Num samples (N) of speech_contribution does not fit to the'
                'shape from speech_prediction',
                self
            )
            assert self.K_target == K_target_, _get_err_msg(
                'Num target speakers (K_target) of speech_contribution does '
                'not fit to the shape from speech_prediction',
                self
            )
            assert self.K_source < 5, _get_err_msg(
                'Num source speakers (K_source) of speech_contribution does '
                'not fit to the shape from speech_source',
                self
            )
            K_target_, samples_ = self.noise_contribution.shape
            assert self.samples == samples_, _get_err_msg(
                'Num samples (N) of noise_contribution does not fit to the '
                'shape from speech_prediction',
                self
            )
            assert self.K_target == K_target_, _get_err_msg(
                'Num target speakers (K_target) of noise_contribution does '
                'not fit to the shape from speech_prediction',
                self
            )
            deviation = np.std(np.abs(
                self.speech_prediction
                - np.sum(self.speech_contribution, axis=0)
                - self.noise_contribution
            ))
            assert deviation < 1e-3, (
                'The deviation of speech prediction and the sum of individual '
                f'contributions is expected to be low: {deviation}'
            )
        else:
            assert (
                self.speech_contribution is None
                and self.noise_contribution is None
            ), (
                'Expect that speech_contribution and noise_contribution are '
                'both None or given.\n'
                'Got:\n'
                f'speech_contribution: {self.speech_contribution}\n'
                f'noise_contribution: {self.noise_contribution}'
            )

    @cached_property.cached_property
    def selection(self):
        return self.mir_eval['selection']

    @cached_property.cached_property
    def speech_prediction_selection(self):
        assert self.speech_prediction.ndim == 2, self.speech_prediction.shape
        assert self.speech_prediction.shape[0] < 10, self.speech_prediction.shape  # NOQA
        assert (
            self.speech_prediction.shape[0]
            in (len(self.selection), len(self.selection) + 1)
        ), self.speech_prediction.shape
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
        try:
            import paderbox as pb
        except ImportError:
            return np.nan

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
        # pypesq does not release the GIL. Either release our pesq code or
        # change pypesq to release the GIL and be thread safe
        try:
            import pypesq
        except ImportError:
            raise AssertionError(
                'To use this pesq implementation, install '
                'https://github.com/ludlows/python-pesq .'
            )
        mode = {8000: 'nb', 16000: 'wb'}[self.sample_rate]

        assert self.speech_source.shape == self.speech_prediction_selection.shape, (self.speech_source.shape, self.speech_prediction_selection.shape)  # NOQA
        assert self.speech_source.ndim == 2, (self.speech_source.shape, self.speech_prediction_selection.shape)  # NOQA
        assert self.speech_source.shape[0] < 5, (self.speech_source.shape, self.speech_prediction_selection.shape)  # NOQA

        return [
            pypesq.pypesq(ref=ref, deg=deg, fs=self.sample_rate, mode=mode)
            for ref, deg in zip(
                self.speech_source, self.speech_prediction_selection)
        ]

    @cached_property.cached_property
    def invasive_sxr(self):
        from pb_bss.evaluation.sxr_module import output_sxr
        invasive_sxr = output_sxr(
            rearrange(
                self.speech_contribution,
                'sources targets samples -> sources targets samples'
            )[:, self.selection, :],
            rearrange(
                self.noise_contribution, 'targets samples -> targets samples'
            )[self.selection, :],
            average_sources=False,
            return_dict=True,
        )
        return invasive_sxr

    @cached_property.cached_property
    def stoi(self):
        return pb_bss.evaluation.stoi(
            reference=self.speech_source,
            estimation=self.speech_prediction_selection,
            sample_rate=self.sample_rate,
        )

    @cached_property.cached_property
    def si_sdr(self):
        if self.enable_si_sdr:
            return pb_bss.evaluation.si_sdr(
                reference=self.speech_source,
                estimation=self.speech_prediction_selection,
            )
        else:
            raise ValueError(
                'SI-SDR is disabled by default since it is only well-defined '
                'for non-reverberant single-channel data. Enable it with '
                '`enable_si_sdr=True`.'
            )

    def as_dict(self):
        metrics = dict(
            pesq=self.pesq,
            stoi=self.stoi,
            mir_eval_sxr_sdr=self.mir_eval['sdr'],
            mir_eval_sxr_sir=self.mir_eval['sir'],
            mir_eval_sxr_sar=self.mir_eval['sar'],
            mir_eval_sxr_selection=self.mir_eval['selection'],
        )

        if self.enable_si_sdr:
            metrics['si_sdr'] = self.si_sdr

        if self._has_contribution_signals:
            metrics['invasive_sxr_sdr'] = self.invasive_sxr['sdr']
            metrics['invasive_sxr_sir'] = self.invasive_sxr['sir']
            metrics['invasive_sxr_snr'] = self.invasive_sxr['snr']

        return metrics
