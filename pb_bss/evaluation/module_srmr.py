import numpy as np


def srmr(signal, sample_rate, n_cochlear_filters=23, low_freq=125, min_cf=4, max_cf=128, fast=False, norm=False,):
    """
    Wrapper around the SRMRpy package to allow an independent axis
    Note: The results of this implementation are slightly different from the Matlab implementation, but a high
    correlation between the behavior of both implementations is still present.
    However, activating the fast implementation or norm drastically changes the absolute values of the results due to
    changes in the gammatone package, is maintained. Please make sure to check the correlation bewteen the
    Matlab implementation and this implementation before activating either.

        >>> import paderbox as pb
        >>> a = pb.testing.testfile_fetcher.get_file_path('speech_bab_0dB.wav')
        >>> a = pb.io.load_audio(a)
        >>> srmr(a, 16000)
        1.865961007729717
        >>> srmr([a, a], 16000, fast=False)
        array([1.86596101, 1.86596101])

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
