import numpy as np


def si_sdr(reference, estimation):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]

    Returns:
        SI-SDR

    SDR– Half- Baked or Well Done?
    http://www.merl.com/publications/docs/TR2019-013.pdf

    >>> np.random.seed(0)
    >>> reference = np.random.randn(100)
    >>> si_sdr(reference, reference)
    325.10914850346956
    >>> si_sdr(reference, reference * 2)
    325.10914850346956
    >>> si_sdr(reference, np.flip(reference))
    -25.127672346460717
    >>> si_sdr(reference, reference + np.flip(reference))
    0.481070445785553
    >>> si_sdr(reference, reference + 0.5)
    6.370460603257728
    >>> si_sdr(reference, reference * 2 + 1)
    6.370460603257728
    >>> si_sdr([1., 0], [0., 0])  # never predict only zeros
    nan
    >>> si_sdr([reference, reference], [reference * 2 + 1, reference * 1 + 0.5])
    array([6.3704606, 6.3704606])

    """
    reference = np.asanyarray(reference)
    estimation = np.asanyarray(estimation)

    assert reference.dtype == np.float64, reference.dtype
    assert estimation.dtype == np.float64, estimation.dtype
    assert reference.shape == estimation.shape, (reference, estimation)

    ref_energy = np.sum(reference ** 2, axis=-1, keepdims=True)
    proj = np.sum(reference * estimation, axis=-1, keepdims=True) * reference / ref_energy
    noise = estimation - proj
    ratio = np.sum(proj ** 2, axis=-1) / (np.sum(noise ** 2, axis=-1))
    si_sdr = 10 * np.log10(ratio)

    return si_sdr
