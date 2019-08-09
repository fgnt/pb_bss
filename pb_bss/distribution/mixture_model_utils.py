import itertools
import numpy as np


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
