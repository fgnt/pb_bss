"""
Run all tests either with:
    nosetests -w tests/
Or with:
    make test
"""

import unittest
import numpy as np
from pb_bss.permutation_alignment import DHTVPermutationAlignment as FPA
from pathlib import Path
import itertools
import pytest


class TestPermutationAlignment(unittest.TestCase):
    def setUp(self):
        pass

    def test_identity_mapping(self):
        estimated_mapping = FPA.get_identity_permutation((2, 3), axis=1)
        reference_mapping = np.asarray(
            [
                [0, 1, 2],
                [0, 1, 2]
            ]
        )
        np.testing.assert_equal(estimated_mapping, reference_mapping)

    def test_local_mapping_on_toy_example(self):
        features = np.asarray(
            [
                [1, 0, 0, 0],
                [0, 0, 2, 0],
                [1, 2, 0, 0],
            ]
        )
        centroids = np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ]
        )
        estimated_mapping = FPA.get_local_mapping(
            features[:, :, None], centroids
        )[0][0, :]
        reference_mapping = [0, 2, 1]
        np.testing.assert_equal(estimated_mapping, reference_mapping)

    def test_inverse_permutation(self):
        permutation = np.asarray([[3, 0, 2, 1]])
        inverse = np.asarray([[1, 3, 2, 0]])
        estimated_inverse = FPA.get_inverse_permutation(permutation)
        np.testing.assert_equal(estimated_inverse, inverse)

    @pytest.mark.flaky(reruns=5)
    def test_toy_example_embedding_based_alignment(self):
        data_dir = Path(__file__).parent
        embedding = np.load(data_dir / 'embedding.npy')
        mask = np.load(data_dir / 'ideal_binary_mask.npy')
        _, E, _ = embedding.shape
        T, K, F = mask.shape

        random_permutation = FPA.random_permutation((F, K))
        permuted_mask = FPA.apply_mapping_to_mask(mask, random_permutation)

        estimated_mask = permuted_mask
        features = FPA.extract_features(estimated_mask, embedding)
        estimated_mask, mapping = FPA.align(estimated_mask, features)

        # Allow global permutation
        mismatch = np.inf
        for global_permutation in itertools.permutations(range(K)):
            mismatch = np.minimum(np.sum(np.abs(
                estimated_mask[:, global_permutation, :] - mask
            )) / mask.size, mismatch)

        np.testing.assert_array_less(mismatch, 0.1)

    def test_parallel_get_local_mapping(self):

        def reference(mapping, features, centroids):
            F = features.shape[2]

            def get_local_mapping(x, mu):
                K = x.shape[0]
                assert K < 10, f'K = {K} seems to be too much.'
                similarity_matrix = np.einsum('ke,le->kl', x, mu)
                best_permutation = None
                best_score = -np.inf
                for permutation in list(itertools.permutations(range(K))):
                    score = np.sum(similarity_matrix[permutation, range(K)])
                    if score > best_score:
                        best_permutation = permutation
                        best_score = score
                return best_permutation, best_score

            total_score = 0
            for f in range(F):
                mapping[f, :], best_score = get_local_mapping(
                    features[:, :, f], centroids
                )
                total_score += best_score
            return mapping, total_score

        K, E, F = 2, 20, 100
        mapping = np.repeat(np.arange(K)[None, :], F, axis=0)
        features = np.random.uniform(size=(K, E, F))
        centroids = np.random.uniform(size=(K, E))

        ref_mapping, ref_total_score = reference(mapping, features, centroids)
        mapping, total_score = FPA.get_local_mapping(features, centroids)

        np.testing.assert_equal(mapping, ref_mapping)
        np.testing.assert_allclose(total_score, ref_total_score, rtol=1e-6)
