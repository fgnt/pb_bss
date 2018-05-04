import typing

import numpy as np


class _Parameter:
    def to_dict(self):
        """
        >>> from IPython.lib.pretty import pprint
        >>> from dc_integration.distribution.cacgmm import (
        ...     ComplexAngularCentralGaussianParameters, 
        ...     ComplexAngularCentralGaussianMixtureModelParameters,
        ... )
        >>> model = ComplexAngularCentralGaussianParameters()
        >>> model
        ComplexAngularCentralGaussianParameters(covariance=None, precision=None, determinant=None)
        >>> pprint(model.to_dict())
        {'covariance': None, 'precision': None, 'determinant': None}
        >>> model = ComplexAngularCentralGaussianMixtureModelParameters()
        >>> model
        ComplexAngularCentralGaussianMixtureModelParameters(cacg=ComplexAngularCentralGaussianParameters(covariance=None, precision=None, determinant=None), mixture_weight=None, affiliation=None, eps=1e-10)
        >>> pprint(model.to_dict())
        {'cacg': {'covariance': None, 'precision': None, 'determinant': None},
         'mixture_weight': None,
         'affiliation': None,
         'eps': 1e-10}

         >>> import jsonpickle, json
         >>> pprint(json.loads(jsonpickle.dumps(model)))
        {'py/object': 'dc_integration.distribution.cacgmm.ComplexAngularCentralGaussianMixtureModelParameters',
         'affiliation': None,
         'cacg': {'py/object': 'dc_integration.distribution.complex_angular_central_gaussian.ComplexAngularCentralGaussianParameters',
          'covariance': None,
          'determinant': None,
          'precision': None},
         'eps': 1e-10,
         'mixture_weight': None}
         >>>
        """
        ret = {
            k: getattr(self, k)
            for k in self.__dataclass_fields__.keys()
        }
        ret = {
            k: v.to_dict() if isinstance(v, _Parameter) else v
            for k, v in ret.items()
        }
        return ret

    @classmethod
    def from_dict(cls, d: dict):
        """

        >>> from IPython.lib.pretty import pprint
        >>> from dc_integration.distribution.cacgmm import (
        ...     ComplexAngularCentralGaussianParameters,
        ...     ComplexAngularCentralGaussianMixtureModelParameters,
        ... )
        >>> model = ComplexAngularCentralGaussianParameters()
        >>> model.determinant = 2
        >>> model
        ComplexAngularCentralGaussianParameters(covariance=None, precision=None, determinant=2)
        >>> d = model.to_dict()
        >>> pprint(d)
        {'covariance': None, 'precision': None, 'determinant': 2}
        >>> ComplexAngularCentralGaussianParameters.from_dict(d)
        ComplexAngularCentralGaussianParameters(covariance=None, precision=None, determinant=2)

        >>> model = ComplexAngularCentralGaussianMixtureModelParameters()
        >>> model.cacg.determinant = 2
        >>> model
        ComplexAngularCentralGaussianMixtureModelParameters(cacg=ComplexAngularCentralGaussianParameters(covariance=None, precision=None, determinant=2), mixture_weight=None, affiliation=None, eps=1e-10)
        >>> d = model.to_dict()
        >>> pprint(d)
        {'cacg': {'covariance': None, 'precision': None, 'determinant': 2},
         'mixture_weight': None,
         'affiliation': None,
         'eps': 1e-10}
        >>> ComplexAngularCentralGaussianMixtureModelParameters.from_dict(d)
        ComplexAngularCentralGaussianMixtureModelParameters(cacg={'covariance': None, 'precision': None, 'determinant': 2}, mixture_weight=None, affiliation=None, eps=1e-10)
        """
        assert cls.__dataclass_fields__.keys() == d.keys(), (cls.__dataclass_fields__.keys(), d.keys())
        return cls(**d)


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


def stack_parameters(parameters: typing.List[_Parameter]):
    """

        >>> from IPython.lib.pretty import pprint
        >>> from dc_integration.distribution.cacgmm import (
        ...     ComplexAngularCentralGaussianParameters,
        ...     ComplexAngularCentralGaussianMixtureModelParameters,
        ... )
        >>> model1 = ComplexAngularCentralGaussianParameters(
        ...     covariance=[1], precision=[2], determinant=[3]
        ... )
        >>> model2 = ComplexAngularCentralGaussianParameters(
        ...     covariance=[3], precision=[4], determinant=[5]
        ... )
        >>> stack_parameters([model1, model2])
        ComplexAngularCentralGaussianParameters(covariance=array([[1],
               [3]]), precision=array([[2],
               [4]]), determinant=array([[3],
               [5]]))

        >>> model3 = ComplexAngularCentralGaussianMixtureModelParameters(
        ...     cacg=model1,
        ...     mixture_weight=[6], affiliation=[7], eps=8
        ... )
        >>> model4 = ComplexAngularCentralGaussianMixtureModelParameters(
        ...     cacg=model2,
        ...     mixture_weight=[9], affiliation=[10], eps=11
        ... )
        >>> stack_parameters([model3, model4])
        ComplexAngularCentralGaussianMixtureModelParameters(cacg=ComplexAngularCentralGaussianParameters(covariance=array([[1],
               [3]]), precision=array([[2],
               [4]]), determinant=array([[3],
               [5]])), mixture_weight=array([[6],
               [9]]), affiliation=array([[ 7],
               [10]]), eps=array([ 8, 11]))

    """
    def get_type(objects):
        types = {p.__class__ for p in objects}
        assert len(types) == 1, types
        return list(types)[0]

    out = get_type(parameters)()

    for k in out.__dataclass_fields__.keys():
        datas = [getattr(p, k) for p in parameters]

        # Ensure unique type
        get_type(datas)

        if hasattr(datas[0], '__dataclass_fields__'):
            data = stack_parameters(datas)
        else:
            data = np.stack(datas)

        setattr(out, k, data)
    return out
