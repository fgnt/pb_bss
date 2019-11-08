# Independent and identically distributed initializers
import numpy as np

__all__ = [
    'uniform_normalized',
    'dirichlet_uniform',
    'dirichlet',
    'one_hot',
]


def uniform_normalized(
        Y,
        num_classes: int,
        permutation_free: bool=False,
):
    """
    affiliation = uniform(num_classes)
    return affiliation / sum(affiliation)

    Args:
        Y: ..., N, D
        sources:
        permutation_free:

    Returns:
        ..., K, N

    >>> np.random.seed(0)
    >>> uniform_normalized(np.ones([4, 5, 3]), 2)
    array([[[0.45937056, 0.62040588, 0.40331128, 0.36119761, 0.52491232],
            [0.54062944, 0.37959412, 0.59668872, 0.63880239, 0.47508768]],
    <BLANKLINE>
           [[0.90086036, 0.96317992, 0.40555365, 0.54326914, 0.0754861 ],
            [0.09913964, 0.03682008, 0.59444635, 0.45673086, 0.9245139 ]],
    <BLANKLINE>
           [[0.60463055, 0.84790293, 0.32818684, 0.59931101, 0.22192973],
            [0.39536945, 0.15209707, 0.67181316, 0.40068899, 0.77807027]],
    <BLANKLINE>
           [[0.2998847 , 0.55847743, 0.4250834 , 0.37590312, 0.0268192 ],
            [0.7001153 , 0.44152257, 0.5749166 , 0.62409688, 0.9731808 ]]])
    >>> uniform_normalized(np.ones([4, 5, 3]), 2, permutation_free=True)
    array([[[0.34898741, 0.67504195, 0.84402017, 0.16032173, 0.64704637],
            [0.65101259, 0.32495805, 0.15597983, 0.83967827, 0.35295363]],
    <BLANKLINE>
           [[0.34898741, 0.67504195, 0.84402017, 0.16032173, 0.64704637],
            [0.65101259, 0.32495805, 0.15597983, 0.83967827, 0.35295363]],
    <BLANKLINE>
           [[0.34898741, 0.67504195, 0.84402017, 0.16032173, 0.64704637],
            [0.65101259, 0.32495805, 0.15597983, 0.83967827, 0.35295363]],
    <BLANKLINE>
           [[0.34898741, 0.67504195, 0.84402017, 0.16032173, 0.64704637],
            [0.65101259, 0.32495805, 0.15597983, 0.83967827, 0.35295363]]])

    """

    independent = Y.shape[:-2]
    num_observations = Y.shape[-2]

    affiliation_shape = (*independent, num_classes, num_observations)

    if permutation_free:
        affiliation = np.random.uniform(size=affiliation_shape[-2:])
        affiliation /= np.einsum("...kn->...n", affiliation)[..., None, :]
        affiliation = np.broadcast_to(affiliation, affiliation_shape)
    else:
        affiliation = np.random.uniform(size=affiliation_shape)
        affiliation /= np.einsum("...kn->...n", affiliation)[..., None, :]
    return affiliation


def dirichlet_uniform(Y, num_classes, permutation_free=False):
    """
    return dirichlet(alpha=1, num_classes)

    Args:
        Y: 
        num_classes: 
        permutation_free: 

    Returns:

    """
    return dirichlet(Y, num_classes, permutation_free, alpha=1)


def dirichlet(
        Y,
        num_classes: int,
        permutation_free: bool=False,
        alpha=1,
):
    """
    return dirichlet(alpha, num_classes)

    Args:
        Y: ..., N, D
        sources:
        permutation_free:

    Returns:
        ..., K, N

    >>> np.random.seed(0)
    >>> dirichlet_uniform(np.ones([4, 5, 3]), 2)
    array([[[0.38788988, 0.53976265, 0.34674414, 0.2056128 , 0.87268651],
            [0.61211012, 0.46023735, 0.65325586, 0.7943872 , 0.12731349]],
    <BLANKLINE>
           [[0.67579094, 0.24418536, 0.44699406, 0.01129788, 0.42463125],
            [0.32420906, 0.75581464, 0.55300594, 0.98870212, 0.57536875]],
    <BLANKLINE>
           [[0.70548556, 0.28983352, 0.109713  , 0.05074518, 0.57941836],
            [0.29451444, 0.71016648, 0.890287  , 0.94925482, 0.42058164]],
    <BLANKLINE>
           [[0.1711358 , 0.42022576, 0.01934878, 0.49670837, 0.71535581],
            [0.8288642 , 0.57977424, 0.98065122, 0.50329163, 0.28464419]]])
    >>> dirichlet_uniform(np.ones([4, 5, 3]), 2, permutation_free=True)
    array([[[0.43676104, 0.95063253, 0.49735576, 0.63117148, 0.45599612],
            [0.56323896, 0.04936747, 0.50264424, 0.36882852, 0.54400388]],
    <BLANKLINE>
           [[0.43676104, 0.95063253, 0.49735576, 0.63117148, 0.45599612],
            [0.56323896, 0.04936747, 0.50264424, 0.36882852, 0.54400388]],
    <BLANKLINE>
           [[0.43676104, 0.95063253, 0.49735576, 0.63117148, 0.45599612],
            [0.56323896, 0.04936747, 0.50264424, 0.36882852, 0.54400388]],
    <BLANKLINE>
           [[0.43676104, 0.95063253, 0.49735576, 0.63117148, 0.45599612],
            [0.56323896, 0.04936747, 0.50264424, 0.36882852, 0.54400388]]])

    """

    independent = Y.shape[:-2]
    num_observations = Y.shape[-2]

    assert np.isscalar(alpha), alpha

    alpha = np.broadcast_to(alpha, (num_classes,))

    if permutation_free:
        affiliation_shape = (*independent, num_classes, num_observations)
        affiliation = np.random.dirichlet(alpha, size=num_observations).T
        affiliation = np.broadcast_to(affiliation, affiliation_shape)
    else:
        affiliation = np.swapaxes(
            np.random.dirichlet(alpha, size=(*independent, num_observations)),
            -1, -2
        )
    return affiliation


def one_hot(
        Y,
        num_classes: int,
        permutation_free: bool=False,
):
    """
    return to_one_hot(multinomial(num_classes))

    Args:
        Y: ..., N, D
        sources:
        permutation_free:

    Returns:
        ..., K, N

    >>> np.random.seed(0)
    >>> one_hot(np.ones([4, 5, 3]), 2)
    array([[[1., 0., 0., 1., 0.],
            [0., 1., 1., 0., 1.]],
    <BLANKLINE>
           [[0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1.]],
    <BLANKLINE>
           [[0., 1., 1., 0., 1.],
            [1., 0., 0., 1., 0.]],
    <BLANKLINE>
           [[1., 1., 1., 1., 0.],
            [0., 0., 0., 0., 1.]]])
    >>> one_hot(np.ones([4, 5, 3]), 2, permutation_free=True)
    array([[[1., 0., 0., 1., 1.],
            [0., 1., 1., 0., 0.]],
    <BLANKLINE>
           [[1., 0., 0., 1., 1.],
            [0., 1., 1., 0., 0.]],
    <BLANKLINE>
           [[1., 0., 0., 1., 1.],
            [0., 1., 1., 0., 0.]],
    <BLANKLINE>
           [[1., 0., 0., 1., 1.],
            [0., 1., 1., 0., 0.]]])

    """

    def label_to_one_hot(labels, number_of_labels):
        """
        Takes as input a label (e.g. 5) and the number of labels (e.g. 10)
        and returns the on hot encoding of that label (e.g. [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        """
        # Original: https://stackoverflow.com/a/42879831/5766934
        return np.eye(number_of_labels)[np.array(labels)]

    independent = Y.shape[:-2]
    num_observations = Y.shape[-2]

    if permutation_free:
        affiliation_shape = (*independent, num_classes, num_observations)
        affiliation = label_to_one_hot(
            np.random.randint(num_classes, size=num_observations),
            num_classes,
        ).T
        affiliation = np.broadcast_to(affiliation, affiliation_shape)
    else:
        affiliation = np.swapaxes(label_to_one_hot(
            np.random.randint(num_classes, size=(*independent, num_observations)),
            num_classes,
        ), -1, -2)
    return affiliation
