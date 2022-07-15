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


class VerboseKeyError(KeyError):
    def __str__(self):
        if len(self.args) == 2:
            item, keys = self.args
            import difflib
            # Suggestions are sorted by their similarity.
            suggestions = difflib.get_close_matches(
                item, keys, cutoff=0, n=100
            )
            return f'{item!r}.\n' \
                   f'Close matches: {suggestions!r}'
        elif len(self.args) == 3:
            item, keys, msg = self.args
            import difflib
            # Suggestions are sorted by their similarity.
            suggestions = difflib.get_close_matches(
                item, keys, cutoff=0, n=100
            )
            return f'{item!r}.\n' \
                   f'Close matches: {suggestions!r}\n' \
                   f'{msg}'
        else:
            return super().__str__()


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
    def mir_eval_sdr(self):
        return self.mir_eval['sdr']

    @cached_property.cached_property
    def mir_eval_sir(self):
        return self.mir_eval['sir']

    @cached_property.cached_property
    def mir_eval_sar(self):
        return self.mir_eval['sar']

    @cached_property.cached_property
    def pesq(self):
        return pb_bss.evaluation.pesq(
                rearrange(
                    [self.speech_source] * self.channels,
                    'channels sources samples -> sources channels samples'
                ),
                [self.observation] * self.K_source,
                sample_rate=self.sample_rate,
        )

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
    def invasive_sdr(self):
        return self.invasive_sxr['sdr']

    @cached_property.cached_property
    def invasive_sir(self):
        return self.invasive_sxr['sir']

    @cached_property.cached_property
    def invasive_snr(self):
        return self.invasive_sxr['snr']

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

    @cached_property.cached_property
    def srmr(self):
        return pb_bss.evaluation.srmr(self.observation, self.sample_rate)

    def _available_metric_names(self):
        metric_names = [
            'pesq',
            'stoi',
            'mir_eval_sdr',
            'mir_eval_sir',
            'mir_eval_sar',
            'srmr'
        ]
        if self.enable_si_sdr:
            metric_names.append('si_sdr')
        if self._has_image_signals:
            metric_names.append('invasive_sdr')
            metric_names.append('invasive_snr')
            metric_names.append('invasive_sir')

        return tuple(metric_names)

    def _disabled_metric_names(self):
        disabled = []
        if not self.enable_si_sdr:
            disabled.append('si_sdr')
        if not self._has_image_signals:
            disabled.append('invasive_sdr')
            disabled.append('invasive_snr')
            disabled.append('invasive_sir')
        return disabled

    def as_dict(self):
        return {name: self[name] for name in self._available_metric_names()}

    def __getitem__(self, item):
        assert isinstance(item, str), (type(item), item)
        try:
            return getattr(self, item)
        except AttributeError:
            pass
        raise VerboseKeyError(
            item,
            self._available_metric_names(),
            f'Disabled: {self._disabled_metric_names()}',
        )


class OutputMetrics:
    def __init__(
            self,
            speech_prediction: 'Shape(K_target, N)',
            speech_source: 'Shape(K_source, N)',
            speech_contribution: 'Shape(K_source, K_target, N)'=None,
            noise_contribution: 'Shape(K_target, N)'=None,
            sample_rate: int = None,
            enable_si_sdr: bool = False,
            compute_permutation: bool = True,
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
            compute_permutation: If True, assume the estimate needs to be
                permuted and use mir_eval_sir to find the permutation. If False,
                assume the estimate has the same permutation as speech_source.

        speech_contribution and noise_contribution can only be calculated for
        linear system and are used for the calculation of invasive_sxr.
        Use speech image (reverberated speech source) and apply for each source
        the enhancement for each target speaker enhancement. The same for the
        noise and each target speaker.

        Example:

            >>> from IPython.lib.pretty import pprint
            >>> metrics = OutputMetrics(
            ...     speech_prediction=np.array([[1., 2., 3., 4.] * 1000,
            ...                                 [4., 3., 2., 1.] * 1000]),
            ...     speech_source=np.array([[1., 2., 2., 3., 2.] * 800,
            ...                             [4., 3., 3., 2., 3.] * 800]),
            ...     sample_rate=8000,
            ... )

            # Obtain all metrics (recommended)
            >>> with np.printoptions(precision=4):
            ...     pprint(metrics.as_dict())
            {'pesq': array([1.2235, 1.225 ]),
             'stoi': array([0.0503, 0.0638]),
             'mir_eval_sdr': array([7.2565, 7.3303]),
             'mir_eval_sir': array([25.6896, 46.638 ]),
             'mir_eval_sar': array([7.3309, 7.3309]),
             'mir_eval_selection': array([0, 1]),
             'srmr': array([125.2507, 126.1846])}

            # Obtain particular metric (e.g. pesq)
            >>> metrics.pesq
            array([1.22345543, 1.2250005 ])

            # Obtain multiple metrics (e.g. pesq and stoi)
            >>> pprint({m: metrics[m] for m in ['pesq', 'stoi']})
            {'pesq': array([1.22345543, 1.2250005 ]),
             'stoi': array([0.05026565, 0.06377457])}
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
        self.compute_permutation = compute_permutation

        self.check_inputs()

    def check_inputs(self):
        assert self.speech_prediction.ndim == 2, self.speech_prediction.shape
        assert self.speech_source.ndim == 2, self.speech_source.shape

        assert self.K_source <= 8, _get_err_msg(
            f'Number of source speakers (K_source) of speech_source is '
            f'{self.K_source}. Expect a reasonable value of 5 or less.',
            self
        )
        assert self.K_target <= 8, _get_err_msg(
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
    def mir_eval_selection(self):
        if self.compute_permutation:
            return self.mir_eval['selection']
        else:
            assert self.K_target == self.K_source, (self.K_target, self.K_source, self.compute_permutation)
            return np.arange(self.K_source)

    @cached_property.cached_property
    def speech_prediction_selection(self):
        assert self.speech_prediction.ndim == 2, self.speech_prediction.shape
        assert self.speech_prediction.shape[0] < 10, self.speech_prediction.shape  # NOQA
        assert (
            self.speech_prediction.shape[0]
            in (len(self.mir_eval_selection), len(self.mir_eval_selection) + 1)
        ), self.speech_prediction.shape
        return self.speech_prediction[self.mir_eval_selection]

    @cached_property.cached_property
    def mir_eval(self):
        return pb_bss.evaluation.mir_eval_sources(
            reference=self.speech_source,
            estimation=self.speech_prediction,
            return_dict=True,
            compute_permutation=self.compute_permutation,
        )

    @cached_property.cached_property
    def mir_eval_sdr(self):
        return self.mir_eval['sdr']

    @cached_property.cached_property
    def mir_eval_sir(self):
        return self.mir_eval['sir']

    @cached_property.cached_property
    def mir_eval_sar(self):
        return self.mir_eval['sar']

    @cached_property.cached_property
    def pesq(self):
        return pb_bss.evaluation.pesq(
            reference=self.speech_source,
            estimation=self.speech_prediction_selection,
            sample_rate=self.sample_rate,
        )

    @cached_property.cached_property
    def invasive_sxr(self):
        from pb_bss.evaluation.sxr_module import output_sxr
        invasive_sxr = output_sxr(
            rearrange(
                self.speech_contribution,
                'sources targets samples -> sources targets samples'
            )[:, self.mir_eval_selection, :],
            rearrange(
                self.noise_contribution, 'targets samples -> targets samples'
            )[self.mir_eval_selection, :],
            average_sources=False,
            return_dict=True,
        )
        return invasive_sxr

    @cached_property.cached_property
    def invasive_sdr(self):
        return self.invasive_sxr['sdr']

    @cached_property.cached_property
    def invasive_sir(self):
        return self.invasive_sxr['sir']

    @cached_property.cached_property
    def invasive_snr(self):
        return self.invasive_sxr['snr']

    @cached_property.cached_property
    def stoi(self):
        return pb_bss.evaluation.stoi(
            reference=self.speech_source,
            estimation=self.speech_prediction_selection,
            sample_rate=self.sample_rate,
        )

    @cached_property.cached_property
    def srmr(self):
        return pb_bss.evaluation.srmr(self.speech_prediction_selection, self.sample_rate)

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

    def _available_metric_names(self):
        metric_names = [
            'pesq',
            'stoi',
            'mir_eval_sdr',
            'mir_eval_sir',
            'mir_eval_sar',
            'mir_eval_selection',
            'srmr',
        ]
        if self.enable_si_sdr:
            metric_names.append('si_sdr')
        if self._has_contribution_signals:
            metric_names.append('invasive_sdr')
            metric_names.append('invasive_snr')
            metric_names.append('invasive_sir')

        return tuple(metric_names)

    def _disabled_metric_names(self):
        disabled = []
        if not self.enable_si_sdr:
            disabled.append('si_sdr')
        if not self._has_contribution_signals:
            disabled.append('invasive_sdr')
            disabled.append('invasive_snr')
            disabled.append('invasive_sir')
        return disabled

    def as_dict(self):
        return {name: self[name] for name in self._available_metric_names()}

    def __getitem__(self, item):
        assert isinstance(item, str), (type(item), item)
        try:
            return getattr(self, item)
        except AttributeError:
            pass
        raise VerboseKeyError(
            item,
            self._available_metric_names(),
            f'Disabled: {self._disabled_metric_names()}',
        )
