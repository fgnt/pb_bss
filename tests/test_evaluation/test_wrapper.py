import numpy as np
import unittest

import pb_bss


class TestMirEval(unittest.TestCase):
    def setUp(self):
        """Variable names are from this signal model:

        \begin{align}
        \mathbf y_td
        &= \mathbf x_t + \mathbf n_t \\
        &= \sum_k \mathbf h_tdk conv s_tk + \mathbf n_td
        \end{align}
        """
        self.channels = channels = 6
        self.speakers = speakers = 2
        self.samples = samples = 8000
        self.sample_rate = 8000

        self.s = np.random.normal(size=(speakers, samples))
        self.x = np.random.normal(size=(speakers, channels, samples))
        self.n = np.random.normal(size=(channels, samples))
        self.y = np.random.normal(size=(channels, samples))
        self.z = np.random.normal(size=(speakers + 1, samples))

        self.speech_contribution \
            = np.random.normal(size=(speakers, speakers + 1, samples))
        self.noise_contribution \
            = np.random.normal(size=(speakers + 1, samples))

        self.z = (
            np.sum(self.speech_contribution, axis=0)
            + self.noise_contribution
        )

        self.input_metrics = pb_bss.evaluation.InputMetrics(
            observation=self.y,
            speech_source=self.s,
            speech_image=self.x,
            noise_image=self.n,
            sample_rate=self.sample_rate,
            enable_si_sdr=True
        )

        self.output_metrics = pb_bss.evaluation.OutputMetrics(
            speech_prediction=self.z,
            speech_source=self.s,
            speech_contribution=self.speech_contribution,
            noise_contribution=self.noise_contribution,
            sample_rate=self.sample_rate,
            enable_si_sdr=True
        )

    def test_input_metrics_mir_eval(self):
        metric = self.input_metrics.mir_eval
        for k, m in metric.items():
            assert m.shape == (self.speakers, self.channels), f'{k}: {m.shape}'

    def test_input_metrics_pesq(self):
        metric = self.input_metrics.pesq
        assert metric.shape == (self.speakers, self.channels), metric.shape

    def test_input_metrics_invasive_sxr(self):
        metric = self.input_metrics.invasive_sxr
        for k, m in metric.items():
            assert m.shape == (self.speakers, self.channels), f'{k}: {m.shape}'

    def test_input_metrics_stoi(self):
        metric = self.input_metrics.stoi
        assert metric.shape == (self.speakers, self.channels), metric.shape

    def test_input_metrics_si_sdr(self):
        metric = self.input_metrics.si_sdr
        assert metric.shape == (self.speakers, self.channels), metric.shape

    def test_output_metrics_mir_eval(self):
        metric = self.output_metrics.mir_eval
        for k, m in metric.items():
            assert m.shape == (self.speakers,), f'{k}: {m.shape}'

    def test_output_metrics_pesq(self):
        metric = self.output_metrics.pesq
        assert metric.shape == (self.speakers,), metric.shape

    def test_output_metrics_invasive_sxr(self):
        metric = self.output_metrics.invasive_sxr
        for k, m in metric.items():
            assert m.shape == (self.speakers,), f'{k}: {m.shape}'

    def test_output_metrics_stoi(self):
        metric = self.output_metrics.stoi
        assert metric.shape == (self.speakers,), metric.shape

    def test_output_metrics_si_sdr(self):
        metric = self.output_metrics.si_sdr
        assert metric.shape == (self.speakers,), metric.shape
