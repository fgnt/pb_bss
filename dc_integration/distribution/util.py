import numpy as np


def _unit_norm(signal, *, axis=-1, eps=1e-4, eps_style='plus'):
    """Unit normalization.

    Args:
        signal: STFT signal with shape (..., T, D).
        eps_style: in ['plus', 'max']
    Returns:
        Normalized STFT signal with same shape.

    >>> signal = np.array([[1, 1], [1e-20, 1e-20], [0, 0]])
    >>> _unit_norm(signal, eps_style='plus')
    array([[7.07056785e-01, 7.07056785e-01],
           [1.00000000e-16, 1.00000000e-16],
           [0.00000000e+00, 0.00000000e+00]])
    >>> _unit_norm(signal, eps_style='max')
    array([[7.07106781e-01, 7.07106781e-01],
           [1.00000000e-16, 1.00000000e-16],
           [0.00000000e+00, 0.00000000e+00]])
    >>> _unit_norm(signal, eps_style='where')  # eps has no effect
    array([[0.70710678, 0.70710678],
           [0.70710678, 0.70710678],
           [0.        , 0.        ]])

    """
    norm = np.linalg.norm(signal, axis=axis, keepdims=True)
    if eps_style == 'plus':
        norm = norm + eps
    elif eps_style == 'max':
        norm = np.maximum(norm, eps)
    elif eps_style == 'where':
        norm = np.where(norm == 0, eps, norm)
    else:
        assert False, eps_style
    return signal / norm
