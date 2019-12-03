import numpy as np
from numpy.testing import assert_allclose
import unittest
from pb_bss.distribution import CACGMMTrainer
from pb_bss.distribution import ComplexAngularCentralGaussian
from pb_bss.distribution import sample_cacgmm
import itertools
from pb_bss.utils import labels_to_one_hot


def solve_permutation(estimated_covariance, covariance):
    K = estimated_covariance.shape[0]

    permutations = list(itertools.permutations(range(K)))
    best_permutation, best_cost = None, np.inf
    for p in permutations:
        cost = np.linalg.norm(estimated_covariance[p, :, :] - covariance)
        if cost < best_cost:
            best_permutation, best_cost = p, cost

    return best_permutation


class TestCACGMM(unittest.TestCase):
    def test_cacgmm(self):
        np.random.seed(0)
        samples = 10000
        weight = np.array([0.3, 0.7])
        covariance = np.array(
            [
                [[10, 1 + 1j, 1 + 1j], [1 - 1j, 5, 1], [1 - 1j, 1, 2]],
                [[2, 0, 0], [0, 3, 0], [0, 0, 2]],
            ]
        )
        covariance /= np.trace(covariance, axis1=-2, axis2=-1)[..., None, None]
        x = sample_cacgmm(samples, weight, covariance)

        model = CACGMMTrainer().fit(
            x,
            num_classes=2,
            covariance_norm='trace',
        )

        # Permutation invariant testing
        best_permutation = solve_permutation(model.cacg.covariance[:, :, :], covariance)

        assert_allclose(
            model.cacg.covariance[best_permutation, :], covariance, atol=0.1
        )

        model.weight = model.weight[best_permutation,]
        assert model.weight[0] < model.weight[1], model.weight
        assert_allclose(model.weight, weight[:, None], atol=0.15)

        # model = CACGMMTrainer().fit(
        #     x,
        #     num_classes=2,
        #     covariance_norm='trace',
        #     dirichlet_prior_concentration=np.inf
        # )
        # assert_allclose(np.squeeze(model.weight, axis=-1), [0.5, 0.5])
        #
        # model = CACGMMTrainer().fit(
        #     x,
        #     num_classes=2,
        #     covariance_norm='trace',
        #     dirichlet_prior_concentration=1_000_000_000
        # )
        # assert_allclose(np.squeeze(model.weight, axis=-1), [0.5, 0.5])

    def test_cacgmm_independent_dimension(self):
        samples = 10000
        weight = np.array([0.3, 0.7])
        covariance = np.array(
            [
                [[10, 1 + 1j, 1 + 1j], [1 - 1j, 5, 1], [1 - 1j, 1, 2]],
                [[2, 0, 0], [0, 3, 0], [0, 0, 2]],
            ]
        )
        covariance /= np.trace(covariance, axis1=-2, axis2=-1)[..., None, None]
        x = sample_cacgmm(samples, weight, covariance)

        model = CACGMMTrainer().fit(
            x[None, ...],
            num_classes=2,
            covariance_norm='trace',
        )

        # Permutation invariant testing
        best_permutation = solve_permutation(model.cacg.covariance[0, :, :, :], covariance)

        assert_allclose(
            np.squeeze(model.weight, axis=(0, -1))[best_permutation,],
            weight,
            atol=0.15
        )
        assert_allclose(
            model.cacg.covariance[0, best_permutation, :], covariance, atol=0.1
        )

        model = CACGMMTrainer().fit(
            np.array([x, x]),
            num_classes=2,
            covariance_norm='trace',
        )

        for f in range(model.weight.shape[0]):
            # Permutation invariant testing
            best_permutation = solve_permutation(model.cacg.covariance[f, :, :, :], covariance)

            assert_allclose(
                np.squeeze(model.weight, axis=-1)[f, best_permutation,],
                weight,
                atol=0.15,
            )
            assert_allclose(
                model.cacg.covariance[f, best_permutation, :],
                covariance,
                atol=0.1,
            )

    def test_cacgmm_sad_init(self):
        samples = 10000
        weight = np.array([0.3, 0.7])
        num_classes, = weight.shape
        covariance = np.array(
            [
                [[10, 1 + 1j, 1 + 1j], [1 - 1j, 5, 1], [1 - 1j, 1, 2]],
                [[2, 0, 0], [0, 3, 0], [0, 0, 2]],
            ]
        )
        covariance /= np.trace(covariance, axis1=-2, axis2=-1)[..., None, None]
        x, labels = sample_cacgmm(samples, weight, covariance, return_label=True)

        affiliations = labels_to_one_hot(labels, num_classes, axis=-2)

        # test initialization
        model = CACGMMTrainer().fit(
            x,
            initialization=affiliations,
            covariance_norm='trace',
        )

        # test initialization with independent
        model = CACGMMTrainer().fit(
            np.array([x]),
            initialization=np.array([affiliations]),
            covariance_norm='trace',
        )

        # test initialization with independent and broadcasted initialization
        model = CACGMMTrainer().fit(
            np.array([x, x, x]),
            initialization=np.array([affiliations]),
            covariance_norm='trace',
        )

        # test initialization with independent
        model = CACGMMTrainer().fit(
            np.array([x, x]),
            initialization=np.array([affiliations, affiliations]),
            covariance_norm='trace',
        )


def test_sample_cacgmm():

    np.random.seed(0)
    samples = 10000
    weight = np.array([0.3, 0.7])
    num_classes = weight.shape[0]
    labels = np.random.choice(
        range(num_classes), size=(samples,), p=weight
    )
    covariance = np.array(
        [
            [[10, 1 + 1j, 1 + 1j], [1 - 1j, 5, 1], [1 - 1j, 1, 2]],
            [[2, 0, 0], [0, 3, 0], [0, 0, 2]],
        ]
    )
    covariance /= np.trace(covariance, axis1=-2, axis2=-1)[..., None, None]
    dimension = covariance.shape[-1]
    x_ref = np.zeros((samples, dimension), dtype=np.complex128)

    for l in range(num_classes):
        cacg = ComplexAngularCentralGaussian.from_covariance(
            covariance=covariance[l, :, :]
        )
        x_ref[labels == l, :] = cacg.sample(size=(np.sum(labels == l),))

    np.random.seed(0)
    x = sample_cacgmm(samples, weight, covariance)

    np.testing.assert_equal(x, x_ref)
