import numpy as np


def stoi(reference, estimation, sample_rate):
    """Wrapper to allow independent axis for STOI.

    Args:
        reference: Shape [..., num_samples]
        estimation: Shape [..., num_samples]
        sample_rate:

    Returns:

    """
    from pystoi.stoi import stoi as pystoi_stoi

    estimation, reference = np.broadcast_arrays(estimation, reference)

    if reference.ndim >= 2:
        return np.array([
            stoi(x_entry, y_entry, sample_rate=sample_rate)
            for x_entry, y_entry in zip(reference, estimation)
        ])
    else:
        return pystoi_stoi(reference, estimation, fs_sig=sample_rate)
