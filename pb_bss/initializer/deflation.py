import numpy as np
from pb_bss.permutation_alignment import _parameterized_vector_norm
import paderbox as pb


def deflationSeed(
        Y,
        sources: int,
        saliencies=None,
        permutation_free: bool=True,
        neighbors: int=5,
        similarity_transform=None,
        eps=0,
):
    """

    Args:
        Y: shape F T D <- TODO: change this to F D T (should match cACGMM)
        sources: number of source
        permutation_free:
        neighbors:

    Returns:
        K, F, T

    """

    if saliencies is None:
        saliencies = np.linalg.norm(Y, axis=-1)
    # F = 257
    # neighbors = 5

    F, T = saliencies.shape
    assert F in [257, 513], F

    Z = _parameterized_vector_norm(Y, axis=-1)

    posterior = []
    for k in range(sources - 1):
        # for f, idx in enumerate(maxidx):

        if permutation_free:
            maxidx = np.argmax(np.mean(saliencies, axis=0), axis=-1)
            maxidx = np.tile(maxidx, F)
        else:
            maxidx = np.argmax(saliencies, axis=-1)

        # Deal with the corners
        maxidx = np.clip(maxidx, neighbors, T - 1 - neighbors)

        Y_local: 'F, D, T_local' = np.stack([
            Y[range(F), maxidx + i, :] for i in
            range(-neighbors, neighbors + 1)
        ], axis=-1)

        saliencies_local: 'F, T_local' = np.stack([
            saliencies[range(F), maxidx + i] for i in
            range(-neighbors, neighbors + 1)
        ], axis=-1)

        psd = pb.speech_enhancement.beamformer.get_power_spectral_density_matrix(
            Y_local, mask=saliencies_local)

        mode = pb.speech_enhancement.beamformer.get_pca_vector(psd)

        similarity = np.abs(np.einsum(
            'FTD,FD->FT',
            Z.conj(),
            _parameterized_vector_norm(mode, axis=-1)
        )) ** 2

        if similarity_transform is not None:
            similarity = similarity_transform(similarity, saliencies)

        posterior.append(similarity)

        distance = 1 - similarity
        saliencies = saliencies * distance

    # The last class captures the rest
    posterior.append(1 - np.sum(posterior, axis=0))

    # The last posterior can be negative. This line fixes this
    posterior = np.maximum(posterior, eps)

    # Ensure the it represents a posterior
    posterior = posterior / np.sum(posterior, axis=0, keepdims=True)

    return posterior
