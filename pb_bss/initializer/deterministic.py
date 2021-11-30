import numpy as np


def flag(
        Y,
        num_classes: int,
        permutation_free: bool=False,
        minimum: float = 0,
):
    """
    Split the time axis is `num_classes` segments and assign each segment to
    one speaker. This is trivial idea that often yield good results [1].
    One concern is, that some models shouldn't be able to recover from the
    zeros [1] and in practice it often still works, because the implementation
    has a flooring [1].
    To overcome this drawback, this function introduce a `minimum` that should
    be between `0` and `1/num_classes` so that each speaker gets at least this
    value and the assigned speaker gets the remaining probability.
    This avoids the numeric oddity, while keeping a good performance.

    This initializsation is inspired by [1], but adds `minimum` and doen't
    split one class to be at the beginning and end.

    Args:
        Y: ..., N, D
        sources:
        permutation_free:

    Returns:
        ..., K, N

    >>> np.random.seed(0)
    >>> flag(np.ones([4, 5, 3]), 2, permutation_free=True)
    array([[[1., 1., 1., 0., 0.],
            [0., 0., 0., 1., 1.]],
    <BLANKLINE>
           [[1., 1., 1., 0., 0.],
            [0., 0., 0., 1., 1.]],
    <BLANKLINE>
           [[1., 1., 1., 0., 0.],
            [0., 0., 0., 1., 1.]],
    <BLANKLINE>
           [[1., 1., 1., 0., 0.],
            [0., 0., 0., 1., 1.]]])
    >>> flag(np.ones([1, 5, 3]), 2, minimum=0.1, permutation_free=True)
    array([[[0.9, 0.9, 0.9, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.9, 0.9]]])
    >>> flag(np.ones([1, 5, 3]), 4, minimum=0.1, permutation_free=True)
    array([[[0.7, 0.7, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.7]]])

    [1] L. Drude and R. Haeb-Umbach, "Integration of Neural Networks and
    Probabilistic Spatial Models for Acoustic Blind Source Separation",
    Ph.D. Thesis

    """
    def label_to_one_hot(labels, number_of_labels):
        """
        Takes as input a label (e.g. 5) and the number of labels (e.g. 10)
        and returns the on hot encoding of that label (e.g. [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        """
        # Original: https://stackoverflow.com/a/42879831/5766934
        return np.eye(number_of_labels)[np.array(labels)]

    if not permutation_free:
        raise NotImplementedError(permutation_free)

    *ind, N, D = Y.shape

    init = np.broadcast_to(label_to_one_hot(np.linspace(
        0, num_classes, N, dtype=int, endpoint=False), num_classes).T,
        [*ind, num_classes, N]
    )
    if minimum == 0:
        pass
    else:
        assert 0 < minimum < (1/num_classes), (minimum, num_classes)

        init = np.maximum(init, minimum/(1-(num_classes-1) * minimum))
        init /= np.sum(init, keepdims=True, axis=-2)
    return init
