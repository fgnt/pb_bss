import typing

import numpy as np


def get_trainer_class_from_model(parameter):
    """
    >>> from IPython.lib.pretty import pprint
    >>> from pb_bss.distribution.cacgmm import (
    ...     ComplexAngularCentralGaussian,
    ... )
    >>> get_trainer_class_from_model(ComplexAngularCentralGaussian).__name__
    'ComplexAngularCentralGaussianTrainer'
    >>> get_trainer_class_from_model(ComplexAngularCentralGaussian()).__name__
    'ComplexAngularCentralGaussianTrainer'

    """
    from pb_bss import distribution

    if not hasattr(parameter, '__name__'):
        parameter = parameter.__class__

    name = parameter.__name__
    assert 'Trainer' not in name, name
    name = name + 'Trainer'

    return getattr(distribution, name)


def _phase_norm(signal, reference_channel=0):
    """Unit normalization.
    Args:
        signal: STFT signal with shape (..., T, D).
    Returns:
        Normalized STFT signal with same shape.
    """
    angles = np.angle(signal[..., [reference_channel]])
    return signal * np.exp(-1j * angles)


def _frequency_norm(
        signal,
        max_sensor_distance=None, shrink_factor=1.2,
        fft_size=1024, sample_rate=16000, sound_velocity=343
):
    """Frequency normalization.
    This function is not really tested, since the use case vanished.
    Args:
        signal: STFT signal with shape (F, T, D).
        max_sensor_distance: Distance in meter.
        shrink_factor: Heuristic shrink factor to move further away from
            the wrapping boarder.
        fft_size:
        sample_rate: In hertz.
        sound_velocity: Speed in meter per second.
    Returns:
        Normalized STFT signal with same shape.
    """
    import paderbox as pb
    frequency = pb.transform.get_stft_center_frequencies(
        fft_size, sample_rate
    )
    # frequency = frequency[1:40]
    F, _, _ = signal.shape
    assert len(frequency) == F
    norm_factor = sound_velocity / (
        2 * frequency * shrink_factor * max_sensor_distance
    )

    # Norm factor can become NaN when one center frequency is zero.
    norm_factor = np.nan_to_num(norm_factor)
    if norm_factor[-1] < 1:
        raise ValueError(
            'Distance between the sensors too high: {:.2} > {:.2}'.format(
                max_sensor_distance, sound_velocity / (2 * frequency[-1])
            )
        )
    norm_factor = norm_factor[:, None, None]
    signal = np.abs(signal) * np.exp(1j * np.angle(signal) * norm_factor)
    return signal


def parameter_from_dict(parameter_class_or_str, d: dict):
    """

    >>> from IPython.lib.pretty import pprint
    >>> from pb_bss.distribution.cacgmm import (
    ...     ComplexAngularCentralGaussian,
    ... )
    >>> model = ComplexAngularCentralGaussian.from_covariance(covariance=[[1]])
    >>> model
    ComplexAngularCentralGaussian(covariance_eigenvectors=array([[1.]]), covariance_eigenvalues=array([1.]))
    >>> d = model.to_dict()
    >>> name = model.__class__.__name__
    >>> pprint(name)
    'ComplexAngularCentralGaussian'
    >>> pprint(d)
    {'covariance_eigenvectors': array([[1.]]),
     'covariance_eigenvalues': array([1.])}
    >>> parameter_from_dict(name, d)
    ComplexAngularCentralGaussian(covariance_eigenvectors=array([[1.]]), covariance_eigenvalues=array([1.]))
    >>> parameter_from_dict(ComplexAngularCentralGaussian, d)
    ComplexAngularCentralGaussian(covariance_eigenvectors=array([[1.]]), covariance_eigenvalues=array([1.]))

    """
    if isinstance(parameter_class_or_str, str):
        from pb_bss import distribution
        # mapping = {
        #     k: getattr(distribution, k)
        #     for k in dir(distribution)
        # }
        # parameter_class_or_str: _Parameter = mapping[parameter_class_or_str]
        parameter_class_or_str = getattr(distribution, parameter_class_or_str)

    return parameter_class_or_str.from_dict(d)


class _ProbabilisticModel:
    def to_dict(self):
        """
        >>> from IPython.lib.pretty import pprint
        >>> from pb_bss.distribution.cacgmm import (
        ...     ComplexAngularCentralGaussian,
        ...     CACGMM,
        ... )
        >>> model = ComplexAngularCentralGaussian()
        >>> model
        ComplexAngularCentralGaussian(covariance_eigenvectors=None, covariance_eigenvalues=None)
        >>> pprint(model.to_dict())
        {'covariance_eigenvectors': None, 'covariance_eigenvalues': None}
        >>> model = CACGMM()
        >>> model
        CACGMM(weight=None, cacg=ComplexAngularCentralGaussian(covariance_eigenvectors=None, covariance_eigenvalues=None))
        >>> pprint(model.to_dict())
        {'weight': None,
         'cacg': {'covariance_eigenvectors': None, 'covariance_eigenvalues': None}}

        >>> import jsonpickle, json
        >>> pprint(json.loads(jsonpickle.dumps(model)))
        {'cacg': {'covariance_eigenvalues': None,
          'covariance_eigenvectors': None,
          'py/object': 'pb_bss.distribution.complex_angular_central_gaussian.ComplexAngularCentralGaussian'},
         'py/object': 'pb_bss.distribution.cacgmm.CACGMM',
         'weight': None}
        """
        keys = self.__dataclass_fields__.keys()
        ret = {
            k: getattr(self, k)
            for k in keys
        }
        ret = {
            k: ret[k].to_dict()
            if isinstance(ret[k], _ProbabilisticModel) else
            ret[k]
            for k in keys
        }
        return ret

    @classmethod
    def from_dict(cls, d: dict):
        """

        >>> from IPython.lib.pretty import pprint
        >>> from pb_bss.distribution.cacgmm import (
        ...     ComplexAngularCentralGaussian,
        ...     CACGMM,
        ... )
        >>> model = ComplexAngularCentralGaussian()
        >>> model.covariance_eigenvectors = 2
        >>> model
        ComplexAngularCentralGaussian(covariance_eigenvectors=2, covariance_eigenvalues=None)
        >>> d = model.to_dict()
        >>> pprint(d)
        {'covariance_eigenvectors': 2, 'covariance_eigenvalues': None}
        >>> ComplexAngularCentralGaussian.from_dict(d)
        ComplexAngularCentralGaussian(covariance_eigenvectors=2, covariance_eigenvalues=None)

        >>> model = CACGMM()
        >>> model.cacg.covariance_eigenvectors = 2
        >>> model
        CACGMM(weight=None, cacg=ComplexAngularCentralGaussian(covariance_eigenvectors=2, covariance_eigenvalues=None))
        >>> d = model.to_dict()
        >>> pprint(d)
        {'weight': None,
         'cacg': {'covariance_eigenvectors': 2, 'covariance_eigenvalues': None}}
        >>> CACGMM.from_dict(d)
        CACGMM(weight=None, cacg={'covariance_eigenvectors': 2, 'covariance_eigenvalues': None})
        """
        assert cls.__dataclass_fields__.keys() == d.keys(), (cls.__dataclass_fields__.keys(), d.keys())
        return cls(**d)

    def __getattr__(self, name):
        """
        >>> from IPython.lib.pretty import pprint
        >>> from pb_bss.distribution.cacgmm import (
        ...     ComplexAngularCentralGaussian,
        ...     CACGMM,
        ... )
        >>> model = ComplexAngularCentralGaussian()
        >>> model.covariances
        Traceback (most recent call last):
        ...
        AttributeError: 'ComplexAngularCentralGaussian' object has no attribute 'covariances'.
        Close matches: ['covariance_eigenvalues', 'covariance_eigenvectors']
        >>> model.abc
        Traceback (most recent call last):
        ...
        AttributeError: 'ComplexAngularCentralGaussian' object has no attribute 'abc'.
        Close matches: ['covariance_eigenvectors', 'covariance_eigenvalues']
        """

        import difflib
        similar = difflib.get_close_matches(name, self.__dataclass_fields__.keys())
        if len(similar) == 0:
            similar = list(self.__dataclass_fields__.keys())

        raise AttributeError(
            f'{self.__class__.__name__!r} object has no attribute {name!r}.\n'
            f'Close matches: {similar}'
        )


def _unit_norm(signal, *, axis=-1, eps=1e-4, eps_style='plus', ord=None):
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
    norm = np.linalg.norm(signal, ord=ord, axis=axis, keepdims=True)
    if eps_style == 'plus':
        norm = norm + eps
    elif eps_style == 'max':
        norm = np.maximum(norm, eps)
    elif eps_style == 'where':
        norm = np.where(norm == 0, eps, norm)
    else:
        assert False, eps_style
    return signal / norm


def stack_parameters(parameters: typing.List[_ProbabilisticModel]):
    """

        >>> from IPython.lib.pretty import pprint
        >>> from pb_bss.distribution.cacgmm import (
        ...     CACGMM,
        ...     ComplexAngularCentralGaussian,
        ... )
        >>> model1 = ComplexAngularCentralGaussian.from_covariance(
        ...     covariance=[[1, 0], [0, 1]]
        ... )
        >>> model2 = ComplexAngularCentralGaussian.from_covariance(
        ...     covariance=[[3, 1], [1, 2]]
        ... )
        >>> stack_parameters([model1, model2])
        ComplexAngularCentralGaussian(covariance_eigenvectors=array([[[ 1.        ,  0.        ],
                [ 0.        ,  1.        ]],
        <BLANKLINE>
               [[ 0.52573111, -0.85065081],
                [-0.85065081, -0.52573111]]]), covariance_eigenvalues=array([[1.        , 1.        ],
               [0.38196601, 1.        ]]))

        >>> model3 = CACGMM(cacg=model1, weight=[6])
        >>> model4 = CACGMM(cacg=model2, weight=[9])
        >>> stack_parameters([model3, model4])
        CACGMM(weight=array([[6],
               [9]]), cacg=ComplexAngularCentralGaussian(covariance_eigenvectors=array([[[ 1.        ,  0.        ],
                [ 0.        ,  1.        ]],
        <BLANKLINE>
               [[ 0.52573111, -0.85065081],
                [-0.85065081, -0.52573111]]]), covariance_eigenvalues=array([[1.        , 1.        ],
               [0.38196601, 1.        ]])))

    """
    def get_type(objects):
        types = {p.__class__ for p in objects}
        assert len(types) == 1, types
        return list(types)[0]

    out_type = get_type(parameters)

    out = {}
    for k in parameters[0].__dataclass_fields__.keys():
        datas = [getattr(p, k) for p in parameters]

        # Ensure unique type
        get_type(datas)

        if hasattr(datas[0], '__dataclass_fields__'):
            data = stack_parameters(datas)
        else:
            data = np.stack(datas)

        # setattr(out, k, data)
        out[k] = data
        # setattr(out, k, data)
    return out_type(**out)


def force_hermitian(matrix):
    """

    >>> A = np.array([[1+2j, 3+5j], [7+11j, 13+17j]])
    >>> force_hermitian(A)
    array([[ 1.+0.j,  5.-3.j],
           [ 5.+3.j, 13.+0.j]])
    >>> force_hermitian(force_hermitian(A))
    array([[ 1.+0.j,  5.-3.j],
           [ 5.+3.j, 13.+0.j]])
    """
    return (matrix + np.swapaxes(matrix.conj(), -1, -2)) / 2
