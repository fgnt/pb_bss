import numpy as np
from functools import partial
import pb_bss


class GriffinLim:
    """
    @article{Griffin1984GriffinLim,
      title={Signal estimation from modified short-time Fourier transform},
      author={Griffin, Daniel and Lim, Jae},
      journal={IEEE Transactions on Acoustics, Speech, and Signal Processing},
      volume={32},
      number={2},
      pages={236--243},
      year={1984},
      publisher={IEEE}
    }

    @article{Gunawan2010MISI,
      title={Iterative phase estimation for the synthesis of separated sources from single-channel mixtures},
      author={Gunawan, David and Sen, Deep},
      journal={IEEE Signal Processing Letters},
      volume={17},
      number={5},
      pages={421--424},
      year={2010},
      publisher={IEEE}
    }
    """
    def __init__(
            self,
            X: 'Shape: (K, T, F)',
            y: 'Shape: (num_samples,)' = None,
            first_guess='istft',
            size=512, shift=128, fading=False,
    ):
        from nara_wpe.utils import stft, istft
        self.stft = partial(
            stft, size=size, shift=shift, fading=fading
        )
        self.istft = partial(
            istft, size=size, shift=shift, fading=fading
        )

        self.X = X
        self.X_dash_dash = X  # Or None?
        self.X_dash = X  # Or None?
        self.y = y

        if first_guess == 'istft':
            self.x_hat = self.istft(X)
        elif first_guess == 'white_gaussian_noise':
            self.x_hat = np.random.randn(size=self.istft(X).shape)
        elif first_guess == 'y':
            K = X.shape[0]
            # Text just under [Gunawan2010MISI] Equation 5
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

        # ToDo: move function get_variance_for_zero_mean_signal to this repo
        from pb_bss.evaluation.sxr_module import get_variance_for_zero_mean_signal

        return dict(
            mir_eval_sdr=np.mean(metrics.mir_eval['sdr']),
            mir_eval_sir=np.mean(metrics.mir_eval['sir']),
            inconsistency=get_variance_for_zero_mean_signal(
                self.X_dash - self.stft(self.istft(self.X_dash))
            )
        )


class MISI(GriffinLim):
    """
    @article{Gunawan2010MISI,
      title={Iterative phase estimation for the synthesis of separated sources from single-channel mixtures},
      author={Gunawan, David and Sen, Deep},
      journal={IEEE Signal Processing Letters},
      volume={17},
      number={5},
      pages={421--424},
      year={2010},
      publisher={IEEE}
    }
    """
    def step(self):
        """
        Do we need power a adjusted version?
        weight = np.sum(np.abs(x_hat) ** 2, axis=-1, keepdims=True) \
            / np.sum(np.abs(x_hat) ** 2)
        x_dash_dash = x_hat + e * weight

        Returns:

        """
        K = self.X.shape[0]

        # [Gunawan2010MISI] Equation 5
        e = self.y - np.sum(self.x_hat, axis=0)

        # [Gunawan2010MISI] Equation 4
        x_dash_dash = self.x_hat + e / K

        self.X_dash_dash = self.stft(x_dash_dash)

        # [Gunawan2010MISI] Equation 3
        self.X_dash = np.abs(self.X) * np.exp(1j * np.angle(self.X_dash_dash))

        # [Gunawan2010MISI] Equation 2
        self.x_hat = self.istft(self.X_dash)
