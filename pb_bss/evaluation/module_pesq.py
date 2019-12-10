import numpy as np


def pesq(reference, estimation, sample_rate, mode=None):
    """Wrapper to allow independent axis for STOI.

    Args:
        reference: Shape [..., num_samples]
        estimation: Shape [..., num_samples]
        sample_rate:

    Returns:

        >>> np.random.seed(0)
        >>> a = np.random.normal(size=16_000)
        >>> b = a + np.random.normal(size=16_000)
        >>> pesq(a, b, sample_rate=16000)
        2.2297563552856445
        >>> pesq(a, b, sample_rate=8000)
        1.0334522724151611
        >>> pesq(a, b, sample_rate=16000, mode='nb')
        3.200247049331665

        >>> pesq([a, a], [b, b], sample_rate=16000)
        array([2.22975636, 2.22975636])
        >>> pesq([[a, a], [b, a]], [[b, b], [b, b]], sample_rate=16000)
        array([[2.22975636, 2.22975636],
               [4.64388847, 2.22975636]])

        >>> pesq(a, b, sample_rate=8000, mode='wb')
        Traceback (most recent call last):
        ...
        AssertionError: ('wb', 8000)
    """

    # pypesq does not release the GIL. Either release our pesq code or
    # change pypesq to release the GIL and be thread safe
    try:
        import pesq
    except ImportError:
        raise AssertionError(
            'To use this pesq implementation, install pesq from\n'
            'https://github.com/ludlows/python-pesq\n'
            'or istall it with `pip install pesq`'
        )

    estimation, reference = np.broadcast_arrays(estimation, reference)

    if mode is None:
        mode = {8000: 'nb', 16000: 'wb'}[sample_rate]
    else:
        if sample_rate == 16000:
            assert mode in ['nb', 'wb'], (mode, sample_rate)
        elif sample_rate == 8000:
            assert mode == 'nb', (mode, sample_rate)
        else:
            raise ValueError(sample_rate)

    assert reference.shape == estimation.shape, (reference.shape, estimation.shape)  # NOQA

    if reference.ndim >= 2:
        for i in range(reference.ndim-1):
            assert reference.shape[i] < 30, (i, reference.shape, estimation.shape)  # NOQA

        return np.array([
            pesq.pesq(
                ref=reference[i],
                deg=estimation[i],
                fs=sample_rate,
                mode=mode,
            )
            for i in np.ndindex(*reference.shape[:-1])
        ]).reshape(reference.shape[:-1])
    elif reference.ndim == 1:
        return pesq.pesq(
            ref=reference, deg=estimation, fs=sample_rate, mode=mode)
    else:
        raise NotImplementedError(reference.ndim)
