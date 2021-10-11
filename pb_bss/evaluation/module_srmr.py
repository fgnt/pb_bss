import numpy as np


def srmr(signal, sample_rate, n_cochlear_filters=23, low_freq=125, min_cf=4, max_cf=128, fast=True, norm=False,):
    """
    Wrapper around the SRMRpy package to allow an independent axis
    Args:
        signal: Time domain signal with Shape [..., num_samples]
        sample_rate:
        n_cochlear_filters: number of cochlear filters per filterbank
        low_freq: start frequency for the SRMR calculation
        min_cf: center frequency of first modulation filter
        max_cf: center frequency of last modulation filter
        fast: boolean, if true uses faster version based on gammatonegram
        norm: boolean, if true uses modulation spectrum energy normalization

        Returns:
            Signal-to-Reverberation Modulation energy ratio

        >>> np.random.seed(0)
        >>> a = np.random.normal(size=16_000)
        >>> srmr(a, 16000)
        0.5799380132880388
        >>> srmr([a, a], 16000)
        array([0.57993801, 0.57993801])
    """

    try:
        import srmrpy
    except ImportError:
        raise AssertionError(
            'To use this srmr implementation, install the SRMRpy package from\n'
            'https://github.com/jfsantos/SRMRpy\n'
        )
    signal = np.asarray(signal)
    if signal.ndim >= 2:
        for i in range(signal.ndim-1):
            assert signal.shape[i] < 30, (i, signal.shape)  # NOQA
        srmrs = []
        for i in np.ndindex(*signal.shape[:-1]):
            # TODO: Add option to also return the SRMR per gammatone filterbank (typically unused in evaluations)
            srmrs.append(srmrpy.srmr(signal[i], sample_rate, n_cochlear_filters=n_cochlear_filters, low_freq=low_freq,
                                     min_cf=min_cf, max_cf=max_cf, fast=fast, norm=norm)[0])
        return np.array(srmrs).reshape(signal.shape[:-1])
    elif signal.ndim == 1:
        # TODO: Add option to also return the SRMR per gammatone filterbank (typically unused in evaluations)
        return srmrpy.srmr(signal, sample_rate, n_cochlear_filters=n_cochlear_filters, low_freq=low_freq,
                           min_cf=min_cf, max_cf=max_cf, fast=fast, norm=norm)[0]
    else:
        raise NotImplementedError(signal.ndim)
