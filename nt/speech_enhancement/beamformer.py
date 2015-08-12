from scipy.linalg import eig
import numpy as np


def gev_beamformer(Y, gamma_n, gamma_x=None, enable_postfilter=False):
    """ A generalized eigenvalue beamformer.

    If `gamma_x` is not specified (e.g. None), it is assumed to be the inverse of `gamma_n`:

    $$\gamma_{x} = \text{min}(\text{max}(1 - \gamma_{n}, 10^{-6}), 1)$$

    :param Y: The mix signal in the format [Time, Microphone, Frequence]
    :param gamma_n: The mask for the noise
    :type gamma_n: numpy.ndarray
    :param gamma_x: The mask for the desired signal
    :type gamma_x: numpy.ndarray
    :return: Complex beamformed signal
    """

    Y = Y.T
    gamma_n = gamma_n.T

    if gamma_x is None:
        gamma_x = np.clip(1 - gamma_n, 1e-6, 1)
    else:
        gamma_x = gamma_x.T

    # Dimensions
    F = Y.shape[0]  # Frequency
    M = Y.shape[1]  # Microphones
    T = Y.shape[2]  # Time

    def calc_phi(Y, mask):
        if mask.ndim == 2:
            mask = mask[:, None, :]
        phi = np.zeros((F, M, M), dtype=np.complex)
        gamma_y = mask * Y
        for f in np.arange(F):
            phi[f, :, :] = np.dot(gamma_y[f, :, :], Y[f, :, :].T.conj())
            phi[f, :, :] = (phi[f, :, :] + phi[f, :, :].T.conj()) / 2
        phi /= np.sum(mask, axis=-1)[:, :, None]
        return phi

    # Covariance matrices
    phi_xx = calc_phi(Y, gamma_x)
    phi_nn = calc_phi(Y, gamma_n)

    # Beamforming vector
    W_gev = np.zeros((F, M), dtype=np.complex)
    for f in range(F):
        eigenvals, eigenvecs = eig(phi_xx[f, :, :], phi_nn[f, :, :])
        W_gev[f, :] = eigenvecs[:, np.argmax(eigenvals)]

    #Postfilter
    if enable_postfilter:
        w_ban = np.zeros(F)
        for f in range(F):
            w_ban[f] = np.sqrt(np.dot(np.dot(np.dot(W_gev[f, :].T.conj(), phi_nn[f]), phi_nn[f]),W_gev[f, :])) / (
                np.dot(np.dot(W_gev[f, :].T.conj(), phi_nn[f]),W_gev[f, :])
            )
        W_gev *= w_ban[:, None]

    # Beamforming signal
    z = np.zeros((F, T), dtype=np.complex)
    for f in range(F):
        z[f, :] = np.dot(W_gev[f].conj(), Y[f, :, :])

    return z.T