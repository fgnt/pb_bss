import itertools
import numpy as np
from pb_bss.permutation_alignment import _PermutationAlignment
from pb_bss.distribution.utils import _unit_norm


def log_pdf_to_affiliation(
        weight,
        log_pdf,
        source_activity_mask=None,
        affiliation_eps=0.,
):
    """

    Args:
        weight: Needs to be broadcast compatible (i.e. unsqueezed).
        log_pdf: Shape (..., K, N)
        source_activity_mask: Shape (..., K, N)
        affiliation_eps:

    Returns:

    """
    # Only check broadcast compatibility
    if source_activity_mask is None:
        _ = np.broadcast_arrays(weight, log_pdf)
    else:
        _ = np.broadcast_arrays(weight, log_pdf, source_activity_mask)

    # The value of affiliation max may exceed float64 range.
    # Scaling (add in log domain) does not change the final affiliation.
    affiliation = log_pdf - np.amax(log_pdf, axis=-2, keepdims=True)

    np.exp(affiliation, out=affiliation)

    # Weight multiplied not in log domain to avoid logarithm of zero.
    affiliation *= weight

    if source_activity_mask is not None:
        assert source_activity_mask.dtype == np.bool, source_activity_mask.dtype  # noqa
        affiliation *= source_activity_mask

    denominator = np.maximum(
        np.sum(affiliation, axis=-2, keepdims=True),
        np.finfo(affiliation.dtype).tiny,
    )
    affiliation /= denominator

    # Strictly, you need re-normalization after clipping. We skip that here.
    if affiliation_eps != 0:
        affiliation = np.clip(
            affiliation, affiliation_eps, 1 - affiliation_eps,
        )

    return affiliation


def log_pdf_to_affiliation_for_integration_models_with_inline_pa(
        weight,
        spatial_log_pdf,
        spectral_log_pdf,
        source_activity_mask=None,
        affiliation_eps=0.
):
    """Inline permutation alignment as in [1] Equation (11) - (12).

    The idea is to reduce disagreement between the spatial and the spectral
    model.

    It is worth knowing that this code can alternatively be realized by a
    tricky application of the permutation solver.

    [1]
    @inproceedings{Drude2018Dual,
        author = {Drude, Lukas and Higuchi, Takuya and Kinoshita, Keisuke and Nakatani, Tomohiro and Haeb-Umbach, Reinhold},
        booktitle = {International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
        organization = {IEEE},
        title = {Dual Frequency- and Block-Permutation Alignment for Deep Learning Based Block-Online Blind Source Separation},
        year = {2018}
    }

    Args:
        weight: Needs to be broadcast compatible (i.e. unsqueezed).
        spatial_log_pdf: Shape (F, K, T)
        spectral_log_pdf: Shape (F, K, T)
        source_activity_mask: Shape (F, K, T)
        affiliation_eps:

    Returns:

    """
    F, num_classes, T = spatial_log_pdf.shape
    permutations = np.asarray(list(itertools.permutations(range(num_classes))))
    affiliation = np.zeros((F, num_classes, T), dtype=np.float64)
    for f in range(F):

        best_permutation = None
        best_auxiliary_function_value = -np.inf
        for permutation in permutations:
            log_pdf = (
                spatial_log_pdf[f, permutation, :] + spectral_log_pdf[f, :, :]
            )

            # The value of affiliation max may exceed float64 range.
            candidate_affiliation \
                = log_pdf - np.max(log_pdf, axis=-2, keepdims=True)
            np.exp(candidate_affiliation, out=candidate_affiliation)
            denominator = np.maximum(
                np.sum(candidate_affiliation, axis=-2, keepdims=True),
                np.finfo(affiliation.dtype).tiny,
            )
            candidate_affiliation /= denominator
            auxiliary_function_value \
                = np.sum(candidate_affiliation * log_pdf, axis=(-2, -1))

            if auxiliary_function_value > best_auxiliary_function_value:
                best_permutation = permutation
                best_auxiliary_function_value = auxiliary_function_value

        affiliation[f, :, :] = log_pdf_to_affiliation(
            np.broadcast_to(weight, spatial_log_pdf.shape)[f, :, :],
            spatial_log_pdf[f, best_permutation, :]
            + spectral_log_pdf[f, :, :],
            source_activity_mask=None
            if source_activity_mask is None
            else source_activity_mask[f, :, :],
            affiliation_eps=affiliation_eps,
        )

    return affiliation


def estimate_mixture_weight(
    affiliation,
    saliency=None,
    weight_constant_axis=-1,
):
    """
    Estimates the mixture weight of a mixture model.

    The simplest version (without saliency and prior):

        return np.mean(affiliation, axis=weight_constant_axis, keepdims=True)

    Args:
        affiliation: Shape: (..., K, T)
        saliency: Shape: (..., K, T)
        weight_constant_axis: int

    Returns:
        mixture weight with the same shape as affiliation, except for the
        weight_constant_axis that is a singleton:
            e.g. for weight_constant_axis == -1: (..., K, 1)
        When the weight_constant_axis is -2 or the positive counterpart,
        then the returned shape is always (K, 1) and the value if 1/K.

    >>> affiliation = [[0.4, 1, 0.4], [0.6, 0, 0.6]]
    >>> estimate_mixture_weight(affiliation)
    array([[0.6],
           [0.4]])
    >>> estimate_mixture_weight(affiliation, weight_constant_axis=-2)
    array([[0.5],
           [0.5]])
    >>> estimate_mixture_weight([affiliation, affiliation])
    array([[[0.6],
            [0.4]],
    <BLANKLINE>
           [[0.6],
            [0.4]]])
    >>> estimate_mixture_weight([affiliation, affiliation], weight_constant_axis=-2)
    array([[0.5],
           [0.5]])
    >>> estimate_mixture_weight([affiliation, affiliation], weight_constant_axis=-3)
    array([[[0.4, 1. , 0.4],
            [0.6, 0. , 0.6]]])

    """
    affiliation = np.asarray(affiliation)

    if isinstance(weight_constant_axis, int) and \
            weight_constant_axis % affiliation.ndim - affiliation.ndim == -2:
        K = affiliation.shape[-2]
        return np.full([K, 1], 1/K)
    elif isinstance(weight_constant_axis, list):
        weight_constant_axis = tuple(weight_constant_axis)

    if saliency is None:
        weight = np.mean(
            affiliation, axis=weight_constant_axis, keepdims=True
        )
    else:
        masked_affiliation = affiliation * saliency[..., None, :]
        weight = _unit_norm(
            np.sum(
                masked_affiliation, axis=weight_constant_axis, keepdims=True
            ),
            ord=1,
            axis=-2,
            eps=1e-10,
            eps_style='where',
        )

    return weight


def _estimate_mixture_weight_with_dirichlet_prior_concentration(
    affiliation,
    saliency=None,
    weight_constant_axis=-1,
    dirichlet_prior_concentration=1,
):
    """
    This function is a starting point for those that want to use a Dirichlet
    prior with a plug-in rule (i.e. MAP estimate instead of MMSE estimate).
    """
    affiliation = np.asarray(affiliation)

    if isinstance(weight_constant_axis, int) and \
            weight_constant_axis % affiliation.ndim - affiliation.ndim == -2:
        K = affiliation.shape[-2]
        return np.full([K, 1], 1/K)

    if saliency is None:
        if dirichlet_prior_concentration == 1:
            weight = np.mean(
                affiliation, axis=weight_constant_axis, keepdims=True
            )
        elif np.isposinf(dirichlet_prior_concentration):
            *independent, K, T = affiliation.shape[-2:]
            weight = np.broadcast_to(1 / K, [*independent, K, 1])
        else:
            assert dirichlet_prior_concentration >= 1, dirichlet_prior_concentration  # noqa
            assert weight_constant_axis == (-1,), (
                'ToDo: implement weight_constant_axis ({}) for '
                'dirichlet_prior_concentration ({}).'
            ).format(weight_constant_axis, dirichlet_prior_concentration)
            # affiliation: ..., K, T
            tmp = np.sum(
                affiliation, axis=weight_constant_axis, keepdims=True
            )
            K, T = affiliation.shape[-2:]

            weight = (
                tmp + (dirichlet_prior_concentration - 1)
             ) / (
                T + (dirichlet_prior_concentration - 1) * K
            )
    else:
        assert dirichlet_prior_concentration == 1, dirichlet_prior_concentration  # noqa
        masked_affiliation = affiliation * saliency[..., None, :]
        weight = _unit_norm(
            np.sum(
                masked_affiliation, axis=weight_constant_axis, keepdims=True
            ),
            ord=1,
            axis=-1,
            eps=1e-10,
            eps_style='where',
        )

    return weight


def apply_inline_permutation_alignment(
        affiliation,
        *,
        quadratic_form=None,
        weight_constant_axis,
        aligner: _PermutationAlignment
):
    """

    Args:
        affiliation: Shape (F, K, T)
        quadratic_form: Exists for cACGMMs, otherwise None. Shape (F, K, T).
        weight_constant_axis: Scalar integer or tuple of scalar integers.
        aligner: A permutation alignment object.

    Returns:

    """
    message = (
        f'Inline permutation alignment reduces mismatch between frequency '
        f'independent mixtures weights and a frequency independent '
        f'observation model. Therefore, we require `affiliation.ndim == 3` '
        f'({affiliation.shape}) and a corresponding '
        f'`weight_constant_axis` ({weight_constant_axis}).'
    )
    assert affiliation.ndim == 3, message
    assert weight_constant_axis in ((-3,), (-3, -1), -3), message

    # F, K, T -> K, F, T
    affiliation = np.transpose(affiliation, (1, 0, 2))
    mapping = aligner.calculate_mapping(affiliation)
    affiliation = aligner.apply_mapping(affiliation, mapping)
    affiliation = np.transpose(affiliation, (1, 0, 2))

    if quadratic_form is not None:
        quadratic_form = np.transpose(quadratic_form, (1, 0, 2))
        quadratic_form = aligner.apply_mapping(quadratic_form, mapping)
        quadratic_form = np.transpose(quadratic_form, (1, 0, 2))

    if quadratic_form is None:
        return affiliation
    else:
        return affiliation, quadratic_form
