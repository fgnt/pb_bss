import itertools
import numpy as np


def mir_eval_sources(
        reference,
        estimation,
        return_dict=False,
        compute_permutation=True,
):
    """

    Reference should contain K speakers, whereas estimated sources should
    contain K or K + 1 estimates. This includes also the noise, to make sure,
    that permutation is calculated correctly, even when noise is confused with
    a speaker.

    :param reference: Time domain signal with shape (K, ..., T)
    :param estimation: Time domain signal
        with shape (K, ..., T) or (K + 1, ..., T)
    :param return_dict:
    :param compute_permutation:
    :return: SXRs ignoring noise reconstruction performance
        with shape (K,), where the dimension is the total number of
        speakers in the source signal.
        The selection has length K, so it tells you which estimated channels
        to pick out of the K + 1 channels, to obtain the K interesting
        speakers.
    """
    from mir_eval.separation import bss_eval_sources as _bss_eval_sources

    if reference.ndim == 2:
        assert reference.ndim == 2, reference.shape
        assert estimation.ndim == 2, estimation.shape
        assert reference.shape[1] == estimation.shape[1], (
            reference.shape,
            estimation.shape,
        )

        if reference.shape == estimation.shape:
            sdr, sir, sar, selection = _bss_eval_sources(
                reference,
                estimation,
                compute_permutation=compute_permutation
            )
        elif reference.shape[0] == estimation.shape[0] - 1:
            if not compute_permutation:
                raise NotImplementedError(compute_permutation, 'with K + 1')
            sdr, sir, sar, selection = _bss_eval_sources_and_noise(
                reference, estimation
            )
        else:
            raise ValueError(
                f"Shapes do not fit: {reference.shape} vs. {estimation.shape}"
            )

    elif reference.ndim >= 3:
        assert reference.shape[1:] == estimation.shape[1:], (
            reference.shape,
            estimation.shape,
        )
        results = np.moveaxis(np.array([
            mir_eval_sources(
                reference[:, d, ..., :],
                estimation[:, d, ..., :],
                compute_permutation=compute_permutation
            )
            for d in range(reference.shape[1])
        ]), source=0, destination=2)
        # D, (sdr, sir, ...), K
        # (sdr, sir, ...), K, D  <- after moveaxis

        if compute_permutation:
            sdr, sir, sar, selection = results
            selection = selection.astype(np.int)
        else:
            sdr, sir, sar = results
            selection = None
    else:
        raise ValueError(f'Strange input shape: {reference.shape}')

    if return_dict:
        if compute_permutation:
            return {"sdr": sdr, "sir": sir, "sar": sar, "selection": selection}
        else:
            return {"sdr": sdr, "sir": sir, "sar": sar}
    else:
        if compute_permutation:
            return sdr, sir, sar, selection
        else:
            return sdr, sir, sar


def _bss_eval_sources_and_noise(reference_sources, estimated_sources):
    """

    Reference should contain K speakers, whereas estimated sources should
    contain K + 1 estimates. This includes also the noise, to make sure, that
    permutation is calculated correctly, even when noise is confused with a
    speaker.

    :param reference_sources: Time domain signal with shape (K, T)
    :param estimated_sources: Time domain signal with shape (K + 1, T)
    :return: SXRs ignoring noise reconstruction performance
        with shape (K,), where the dimension is the total number of
        speakers in the source signal.
        The selection has length K, so it tells you which estimated channels
        to pick out of the K + 1 channels, to obtain the K interesting
        speakers.
    """
    from mir_eval.separation import _bss_decomp_mtifilt
    from mir_eval.separation import _bss_source_crit
    K, T = reference_sources.shape
    assert estimated_sources.shape == (K + 1, T), estimated_sources.shape

    # Compute criteria for all possible pair matches
    sdr = np.empty((K + 1, K))
    sir = np.empty((K + 1, K))
    sar = np.empty((K + 1, K))

    for j_est in range(K + 1):
        for j_true in range(K):
            s_true, e_spat, e_interf, e_artif = _bss_decomp_mtifilt(
                reference_sources, estimated_sources[j_est], j_true, 512
            )
            sdr[j_est, j_true], sir[j_est, j_true], sar[
                j_est, j_true
            ] = _bss_source_crit(s_true, e_spat, e_interf, e_artif)

    # Select the best ordering, while ignoring the noise reconstruction.
    permutations = list(itertools.permutations(list(range(K + 1)), K))

    mean_sir = np.empty(len(permutations))
    dum = np.arange(K)
    for (i, permutation) in enumerate(permutations):
        mean_sir[i] = np.mean(sir[permutation, dum])

    optimal_selection = permutations[np.argmax(mean_sir)]
    idx = (optimal_selection, dum)

    return sdr[idx], sir[idx], sar[idx], np.asarray(optimal_selection)
