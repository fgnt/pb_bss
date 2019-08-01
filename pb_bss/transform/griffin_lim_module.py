import numpy as np
import paderbox as pb
from functools import partial
import pb_bss
from paderbox.speech_enhancement.noise import get_variance_for_zero_mean_signal


class GriffinLim:
    def __init__(
            self,
            X,
            y=None,
            first_guess='istft',
            size=512, shift=128, fading=False,
    ):
        self.stft = partial(
            pb.transform.stft, size=size, shift=shift, fading=fading
        )
        self.istft = partial(
            pb.transform.istft, size=size, shift=shift, fading=fading
        )

        self.X = X
        self.X_dash_dash = X  # Or None?
        self.X_dash = X  # Or None?
        self.y = y

        if first_guess == 'istft':
            self.x_hat = self.istft(X)
        elif first_guess == 'y':
            K = X.shape[0]
            self.x_hat = np.repeat(self.y[None, :] / K, K, axis=0)
        else:
            raise ValueError(first_guess)

    def step(self):
        self.X_dash_dash = self.stft(self.x_hat)
        self.X_dash = np.abs(self.X) * np.exp(1j * np.angle(self.X_dash_dash))
        self.x_hat = self.istft(self.X_dash)

    def evaluate(self, speech_source):
        """

        Args:
            speech_source: Oracle for evaluation.

        Returns:

        """
        metrics = pb_bss.evaluation.OutputMetrics(
            speech_prediction=self.x_hat,
            speech_source=speech_source,
            enable_si_sdr=True,
        )
        return dict(
            mir_eval_sdr=np.mean(metrics.mir_eval['sdr']),
            mir_eval_sir=np.mean(metrics.mir_eval['sir']),
            inconsistency=get_variance_for_zero_mean_signal(
                self.X_dash - self.stft(self.istft(self.X_dash))
            )
        )


class MISI(GriffinLim):
    def step(self):
        """
        Do we need power a adjusted version?
        weight = np.sum(np.abs(x_hat) ** 2, axis=-1, keepdims=True) \
            / np.sum(np.abs(x_hat) ** 2)
        x_dash_dash = x_hat + e * weight

        Returns:

        """
        K = self.X.shape[0]
        e = self.y - np.sum(self.x_hat, axis=0)
        x_dash_dash = self.x_hat + e / K
        self.X_dash_dash = self.stft(x_dash_dash)
        self.X_dash = np.abs(self.X) * np.exp(1j * np.angle(self.X_dash_dash))
        self.x_hat = self.istft(self.X_dash)
