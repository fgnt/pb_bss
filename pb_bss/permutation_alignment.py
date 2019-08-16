import numpy as np
import itertools

__all__ = [
    'DHTVPermutationAlignment',
    'OraclePermutationAlignment',
    'GreedyPermutationAlignment',
]


def interleave(*lists):
    """ Interleave multiple lists. Input does not need to be of equal length.

    based on http://stackoverflow.com/a/29566946/911441

    >>> a = [1, 2, 3, 4, 5]
    >>> b = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    >>> list(interleave(a, b))
    [1, 'a', 2, 'b', 3, 'c', 4, 'd', 5, 'e', 'f', 'g']
    >>> list(interleave(b, a))
    ['a', 1, 'b', 2, 'c', 3, 'd', 4, 'e', 5, 'f', 'g']

    Args:
        lists: An arbitrary number of lists

    Returns: Interleaved lists

    """
    iterators = [iter(l) for l in lists]
    while True:
        for iter_idx in range(len(iterators)):
            try:
                if iterators[iter_idx] is not None:
                    yield next(iterators[iter_idx])
            except StopIteration:
                iterators[iter_idx] = None
        if all(i is None for i in iterators):
            break


def sample_random_mapping(K, F, random_state=np.random):
    """Generate random mapping.

    Args:
        K: Speakers/ sources/ mixture components.
        F: Frequency bins.
        random_state: Numpy random state. Defaults to `np.random`.
    Returns:
        Random mapping with shape (K, F).
    """
    return np.stack([random_state.permutation(K) for f in range(F)], axis=1)


def apply_mapping(mask, mapping):
    """Applies the mapping to obtain a frequency aligned mask.

    Args:
        mask: Permuted mask with shape (K, F, ...).
        mapping: Reverse mapping with shape (K, F).

    >>> np.random.seed(0)
    >>> K, F, T = 3, 5, 6
    >>> reference_mask = np.zeros((K, F, T), dtype=np.int8)
    >>> def viz_mask(mask: np.ndarray):
    ...     mask = np.einsum('KFT->FKT', mask).astype(str).tolist()
    ...     for mask_f in mask:
    ...         print('   '.join([' '.join(m) for m in mask_f]))
    >>> reference_mask[0, :, :2] = 1
    >>> reference_mask[1, :, 2:4] = 1
    >>> reference_mask[2, :, 4:] = 1
    >>> viz_mask(reference_mask)
    1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
    1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
    1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
    1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
    1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
    >>> mapping = sample_random_mapping(K, F)
    >>> mask = apply_mapping(reference_mask, mapping)
    >>> viz_mask(mask)
    0 0 0 0 1 1   0 0 1 1 0 0   1 1 0 0 0 0
    0 0 0 0 1 1   1 1 0 0 0 0   0 0 1 1 0 0
    1 1 0 0 0 0   0 0 0 0 1 1   0 0 1 1 0 0
    0 0 0 0 1 1   1 1 0 0 0 0   0 0 1 1 0 0
    0 0 0 0 1 1   0 0 1 1 0 0   1 1 0 0 0 0

    Test against a loopy implementation of apply mapping
    >>> def apply_mapping_loopy(mask, mapping):
    ...     _, F = mapping.shape
    ...     aligned_mask = np.zeros_like(mask)
    ...     for f in range(F):
    ...         aligned_mask[:, f, :] = mask[mapping[:, f], f, :]
    ...     return aligned_mask
    >>> mask = apply_mapping_loopy(reference_mask, mapping)
    >>> viz_mask(mask)
    0 0 0 0 1 1   0 0 1 1 0 0   1 1 0 0 0 0
    0 0 0 0 1 1   1 1 0 0 0 0   0 0 1 1 0 0
    1 1 0 0 0 0   0 0 0 0 1 1   0 0 1 1 0 0
    0 0 0 0 1 1   1 1 0 0 0 0   0 0 1 1 0 0
    0 0 0 0 1 1   0 0 1 1 0 0   1 1 0 0 0 0
    """
    K, F = mapping.shape
    assert K < 20, (K, mapping.shape)
    assert mask.shape[:2] == mapping.shape, (mask.shape, mapping.shape)
    return mask[mapping, range(F)]


class _PermutationAlignment:

    def calculate_mapping(self, mask, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, mask, *args, **kwargs):
        """Calculates mapping and applies it to the provided mask.

        Args:
            mask: Permuted mask with shape (K, F, T).

        """
        mapping = self.calculate_mapping(mask, *args, **kwargs)
        return self.apply_mapping(mask, mapping)

    @staticmethod
    def apply_mapping(mask, mapping):
        """Applies the mapping to obtain a frequency aligned mask.

        Args:
            mask: Permuted mask with shape (K, F, T).
            mapping: Reverse mapping with shape (K, F).
        """
        return apply_mapping(mask, mapping)


class DHTVPermutationAlignment(_PermutationAlignment):
    """Encapsulates all frequency permutation related functions.

    The main algorithm is an implementation of the unpublished frequency
    permutation alignment algorithm from [TranVu2015BSS].

    This does not solve the global permutation problem.
    """
    def __init__(
            self, *,
            stft_size,
            segment_start, segment_width, segment_shift,
            main_iterations, sub_iterations,
            similarity_metric='cos',
            algorithm='greedy',
    ):
        self.stft_size = stft_size
        self.segment_start = segment_start
        self.segment_width = segment_width
        self.segment_shift = segment_shift
        self.main_iterations = main_iterations
        self.sub_iterations = sub_iterations
        self.similarity_metric = similarity_metric
        self.algorithm = algorithm

        # Use faster implementation for cos
        self.get_score_matrix = getattr(
            _ScoreMatrix,
            {'cos': 'multiply'}.get(similarity_metric, similarity_metric),
        )

    @classmethod
    def from_stft_size(cls, stft_size, similarity_metric='cos'):
        """ Use this, if you do not want to set all parameters manually."""
        if stft_size == 512:
            return cls(
                stft_size=stft_size,
                segment_start=70, segment_width=100, segment_shift=20,
                main_iterations=20, sub_iterations=2,
                similarity_metric=similarity_metric,
            )
        elif stft_size == 1024:
            return cls(
                stft_size=stft_size,
                segment_start=100, segment_width=100, segment_shift=20,
                main_iterations=20, sub_iterations=2,
                similarity_metric=similarity_metric,
            )
        else:
            raise ValueError(
                'There is no default for stft_size={}.', stft_size
            )

    def _align_segment(self, mask, prototype):
        """Calculates permutation for a single frequency.
        This algorithm is greedy and finds the suboptimal solution that is
        often similar good as the optimal solution.

        An example can be found in [Boeddecker2015Free] on page 33/ 34.

        Args:
            prototype: Often called prototype or centroid with shape (K, T).
            mask: The permuted mask for the given frequency, shape (K, T).
        Returns:
            Reverse permutation.
        """
        K, T = prototype.shape
        assert K < 10, (K, 'Sure?')
        c_matrix = self.get_score_matrix(mask, prototype)
        return _mapping_from_score_matrix(c_matrix, algorithm=self.algorithm)

    @property
    def alignment_plan(self):
        """Provides the alignment plan for a given setup. Nice to plot, too.
        >>> from pb_bss.permutation_alignment import DHTVPermutationAlignment
        >>> import matplotlib.pyplot as plt
        >>> alignment_plan = DHTVPermutationAlignment.from_stft_size(512).alignment_plan

        # >>> fig, ax = plt.subplots()
        # >>> for i, s in enumerate(alignment_plan):
        # ...     _ = ax.broken_barh([[s[1], s[2] - s[1]]] , (i - 0.25, 0.5))
        # >>> _ = ax.set_xlabel('Frequency bin')
        # >>> _ = ax.set_ylabel('Iteration')
        # >>> plt.show()

        >>> from IPython.lib.pretty import pprint
        >>> pprint(alignment_plan)
        [[20, 70, 170],
         [2, 90, 190],
         [2, 50, 150],
         [2, 110, 210],
         [2, 30, 130],
         [2, 130, 230],
         [2, 0, 110],
         [2, 150, 257]]
        >>> DHTVPermutationAlignment(stft_size=512,
        ...     segment_start=70, segment_width=300, segment_shift=20,
        ...     main_iterations=20, sub_iterations=2).alignment_plan
        Traceback (most recent call last):
        ...
        ValueError: segment_start (70) + segment_width (300)
        must be smaller than stft_size // 2 + 1 (257),
        but it is 370
        >>> DHTVPermutationAlignment(stft_size=512,
        ...     segment_start=0, segment_width=257, segment_shift=20,
        ...     main_iterations=20, sub_iterations=2).alignment_plan
        [[20, 0, 257]]
        """
        F = self.stft_size // 2 + 1

        if self.segment_start + self.segment_width > F:
            raise ValueError(
                f'segment_start ({self.segment_start}) '
                f'+ segment_width ({self.segment_width})\n'
                f'must be smaller than stft_size // 2 + 1 ({F}),\n'
                f'but it is {self.segment_start + self.segment_width}'
            )

        alignment_plan_lower_start = [
            [
                self.sub_iterations,
                segment_start,
                segment_start + self.segment_width,
            ]
            for segment_start in range(
                self.segment_start + self.segment_shift,
                F - self.segment_width, self.segment_shift
            )
        ]
        alignment_plan_higher_start = [
            [
                self.sub_iterations,
                segment_start,
                segment_start + self.segment_width,
            ]
            for segment_start in range(
            self.segment_start - self.segment_shift, 0, -self.segment_shift
        )]

        first_alignment_plan = [
            self.main_iterations,
            self.segment_start,
            self.segment_start + self.segment_width
        ]

        # fix for first and last segment
        if len(alignment_plan_lower_start) > 0:
            alignment_plan_lower_start[-1][-1] = F
        else:
            first_alignment_plan[-1] = F
        if len(alignment_plan_higher_start) > 0:
            alignment_plan_higher_start[-1][1] = 0
        else:
            first_alignment_plan[1] = 0

        alignment_plan_start = list(interleave(
            alignment_plan_lower_start, alignment_plan_higher_start
        ))

        alignment_plan = [first_alignment_plan] + alignment_plan_start
        return alignment_plan

    def calculate_mapping(self, mask, plot=False):
        """Returns just the mapping based on permuted mask input.

        Args:
            mask: Permuted mask with shape (K, F, T).
        Returns:
            Reverse mapping with shape (K, F).
        """
        K, F, _ = mask.shape

        assert F % 2 == 1, (F, 'Sure? Usually F is odd.')

        # (K, F, T)
        # features = mask / np.linalg.norm(mask, axis=-1, keepdims=True)
        if self.similarity_metric in ['cos']:
           features = _parameterized_vector_norm(mask, axis=-1)
        else:
            features = mask.copy()

        mapping = np.repeat(np.arange(K)[:, None], F, axis=1)

        if plot:
            from paderbox import visualization as vis
            # visualization import plot, axes_context

            tmp_mask = apply_mapping(features, mapping)

            with vis.axes_context(K + 1) as axes:
                for tmp_mask_k in tmp_mask:
                    vis.plot.mask(
                        tmp_mask_k.T,
                        ax=axes.new,
                        title=f'start',
                    )
                vis.plot.mask(mapping, limits=None, ax=axes.new)

        for iterations, start, end in self.alignment_plan:
            for iteration in range(iterations):
                # (K, T)
                time_centroid = np.mean(features[:, start:end, :], axis=1)

                if self.similarity_metric in ['cos']:
                    time_centroid = _parameterized_vector_norm(
                        time_centroid,
                        axis=-1,
                    )

                nothing_changed = True
                for f in range(start, end):
                    reverse_permutation = self._align_segment(
                        features[:, f, :], time_centroid,
                    )
                    if not (reverse_permutation == list(range(K))).all():
                        nothing_changed = False
                        features[:, f, :] = features[reverse_permutation, f, :]
                        mapping[:, f] = mapping[reverse_permutation, f]

                if nothing_changed:
                    break

        return mapping


def _parameterized_vector_norm(
        a,
        axis=-1,
):
    """
    Calculates a normalized vector.

    When the values of the input vector are zero, then the returned vector is
    also zero.

    >>> a = np.array([4, 3])
    >>> _parameterized_vector_norm(a)
    array([0.8, 0.6])
    >>> a = np.array([0, 0])
    >>> _parameterized_vector_norm(a)
    array([0., 0.])
    """
    norm = np.linalg.norm(a, axis=axis, keepdims=True)
    tiny = np.finfo(norm.dtype).tiny
    return a / np.maximum(norm, tiny)


class _ScoreMatrix:
    """
    This class is a namespace for functions that return a score matix.

    Args:
        mask:
            shape: sources (K), ..., time (T)
        reference_mask:
            shape: sources (K), ..., time (T)

    Returns:
        score_matrix: input for _mapping_from_score_matrix
            shape: ..., sources (K), sources (K)


    """
    @classmethod
    def cos(cls, mask, reference_mask):
        return cls.multiply(
            _parameterized_vector_norm(mask, axis=-1),
            _parameterized_vector_norm(reference_mask, axis=-1),
        )

    @classmethod
    def multiply(cls, mask, reference_mask):
        score_matrix = np.einsum(
            'K...T,k...T->...kK',
            mask.conj(),
            reference_mask,
        )
        return score_matrix

    @classmethod
    def euclidean(cls, mask, reference_mask):
        # Note: the minus converts the distance to a similarity
        score_matrix = -np.sqrt(np.sum(
            np.abs(mask[:, None, ...] - reference_mask[None, ...]) ** 2,
            axis=-1
        )).T
        return score_matrix

    @classmethod
    def from_name(cls, similarity_metric):
        """
        >>> _ScoreMatrix.from_name('coss')
        Traceback (most recent call last):
        ...
        AttributeError: type object '_ScoreMatrix' has no attribute 'coss'
        Suggestions: cos, euclidean, from_name, multiply

        """
        try:
            return getattr(cls, similarity_metric)
        except AttributeError as e:
            attrs = ', '.join([
                a
                for a in dir(cls)
                if not (
                        a.startswith('__')
                        or a.endswith('__')
                        or a == 'similarity_metric'
                )
            ])
            raise AttributeError(str(e) + '\nSuggestions: ' + attrs) from e


def _calculate_score_matrix(mask, reference_mask, similarity_metric):
    if np.iscomplexobj(mask) or np.iscomplexobj(reference_mask):
        raise NotImplementedError(mask.dtype, reference_mask.dytpe)

    if similarity_metric in ['cos']:
        mask = _parameterized_vector_norm(mask, axis=-1)
        reference_mask = _parameterized_vector_norm(reference_mask, axis=-1)

    if similarity_metric in ['cos', 'multiply']:
        score_matrix = np.einsum('K...T,k...T->...kK', mask.conj(),
                                 reference_mask)
    elif similarity_metric in ['euclidean']:
        # Note: the minus converts the distance to a similarity
        score_matrix = -np.rollaxis(np.sqrt(np.sum(
            np.abs(mask[None, ...] - reference_mask[:, None, ...]) ** 2,
            axis=-1
        )), axis=-1)
    else:
        raise ValueError(similarity_metric)

    return score_matrix


def _mapping_from_score_matrix(score_matrix, algorithm='optimal'):
    """

    The example is chosen such, that `optimal` and `greedy` produce different
    solutions.

    >>> score_matrix = np.array([[11, 10, 0],[4, 5, 10],[6, 0, 5]])
    >>> score_matrix
    array([[11, 10,  0],
           [ 4,  5, 10],
           [ 6,  0,  5]])
    >>> permutation = _mapping_from_score_matrix(score_matrix, 'optimal')
    >>> score_matrix[range(3), permutation]
    array([10, 10,  6])
    >>> permutation = _mapping_from_score_matrix(score_matrix, 'greedy')
    >>> score_matrix[range(3), permutation]
    array([11, 10,  0])


    >>> _mapping_from_score_matrix(score_matrix, 'greedy')
    array([0, 2, 1])
    >>> _mapping_from_score_matrix([score_matrix, score_matrix], 'greedy')
    array([[0, 0],
           [2, 2],
           [1, 1]])
    >>> _mapping_from_score_matrix([score_matrix, score_matrix], 'optimal')
    array([[1, 1],
           [2, 2],
           [0, 0]])

    """
    score_matrix = np.asanyarray(score_matrix)

    *F, K, K_ = score_matrix.shape
    assert K == K_, (score_matrix.shape, K, K_)

    if algorithm in ['greedy']:
        # Example score matrix and selected permutation (#):
        #  11# 10   0
        #   4   5  10#
        #   6   0#  5

        reverse_permutation = np.zeros((K, *F), dtype=np.int)
        # estimated_permutation = np.zeros((K,), dtype=np.int)

        score_matrix: np.ndarray = score_matrix.copy()
        # score_matrix_flat is a view in score_matrix
        # -> changing score_matrix also changes score_matrix_flat
        score_matrix_flat = score_matrix.reshape(*F, K*K)

        for f in np.ndindex(*F):
            for _ in range(K):
                # argmax does not support axis=(-2, -1)
                # -> use score_matrix_flat
                i, j = np.unravel_index(
                    np.argmax(score_matrix_flat[f], axis=-1),
                    score_matrix[f].shape,
                )
                score_matrix[(*f, i, slice(None))] = float('-inf')
                score_matrix[(*f, slice(None), j)] = float('-inf')

                reverse_permutation[(i, *f)] = j
                # estimated_permutation[j] = i

        mapping = reverse_permutation

    elif algorithm in ['optimal']:
        # Example score matrix and selected permutation (#):
        #  11  10#  0
        #   4   5  10#
        #   6#  0   5

        mapping = np.zeros((K, *F), dtype=np.int)

        for f in np.ndindex(*F):
            best_score = float('-inf')
            best_permutation = None
            for permutation in itertools.permutations(range(K)):
                score = sum(score_matrix[(*f, range(K), permutation)])
                if score > best_score:
                    best_score = score
                    best_permutation = permutation
            mapping[(slice(None), *f)] = best_permutation

    else:
        raise ValueError(algorithm)
    return mapping


class GreedyPermutationAlignment(_PermutationAlignment):
    def __init__(
            self,
            similarity_metric='euclidean',
            algorithm='optimal',
    ):
        """
        Calculates a greedy mapping to solve the permutation problem.
        Calculates between adjacent frequencies the `similarity_metric` and
        from that matrix the optimal permutation (`algorithm='optimal'`) or a
        greedy solution (`algorithm='greedy'`, see _mapping_from_score_matrix)

        Args:
            similarity_metric:
            algorithm:
        """
        try:
            self.get_score_matrix = getattr(_ScoreMatrix, similarity_metric)
        except Exception:
            raise ValueError(similarity_metric)
        self.algorithm = algorithm

    def calculate_mapping(self, mask):
        """

        The time frame dimension is interpreted as vector dimension.
        The frequency dimension is interpreted as independent dimension.
        The sources dimension is interpreted as permuted dimension.

        Args:
            mask:
                shape: sources (K), frequencies (F), time frames (T)

        Returns:
            mapping:
                shape: sources (K), frequencies (F)

        >>> np.random.seed(0)
        >>> K, F, T = 3, 5, 6
        >>> reference_mask = np.zeros((K, F, T), dtype=np.int8)
        >>> def viz_mask(mask: np.ndarray):
        ...     mask = np.einsum('KFT->FKT', mask).astype(str).tolist()
        ...     for mask_f in mask:
        ...         print('   '.join([' '.join(m) for m in mask_f]))
        >>> viz_mask(reference_mask)
        0 0 0 0 0 0   0 0 0 0 0 0   0 0 0 0 0 0
        0 0 0 0 0 0   0 0 0 0 0 0   0 0 0 0 0 0
        0 0 0 0 0 0   0 0 0 0 0 0   0 0 0 0 0 0
        0 0 0 0 0 0   0 0 0 0 0 0   0 0 0 0 0 0
        0 0 0 0 0 0   0 0 0 0 0 0   0 0 0 0 0 0
        >>> reference_mask[0, :, :2] = 1
        >>> reference_mask[1, :, 2:4] = 1
        >>> reference_mask[2, :, 4:] = 1
        >>> viz_mask(reference_mask)
        1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
        1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
        1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
        1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
        1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
        >>> mask = apply_mapping(reference_mask, sample_random_mapping(K, F))
        >>> viz_mask(mask)
        0 0 0 0 1 1   0 0 1 1 0 0   1 1 0 0 0 0
        0 0 0 0 1 1   1 1 0 0 0 0   0 0 1 1 0 0
        1 1 0 0 0 0   0 0 0 0 1 1   0 0 1 1 0 0
        0 0 0 0 1 1   1 1 0 0 0 0   0 0 1 1 0 0
        0 0 0 0 1 1   0 0 1 1 0 0   1 1 0 0 0 0
        >>> viz_mask(GreedyPermutationAlignment('cos')(mask))
        0 0 0 0 1 1   0 0 1 1 0 0   1 1 0 0 0 0
        0 0 0 0 1 1   0 0 1 1 0 0   1 1 0 0 0 0
        0 0 0 0 1 1   0 0 1 1 0 0   1 1 0 0 0 0
        0 0 0 0 1 1   0 0 1 1 0 0   1 1 0 0 0 0
        0 0 0 0 1 1   0 0 1 1 0 0   1 1 0 0 0 0
        >>> viz_mask(GreedyPermutationAlignment('euclidean')(mask))
        0 0 0 0 1 1   0 0 1 1 0 0   1 1 0 0 0 0
        0 0 0 0 1 1   0 0 1 1 0 0   1 1 0 0 0 0
        0 0 0 0 1 1   0 0 1 1 0 0   1 1 0 0 0 0
        0 0 0 0 1 1   0 0 1 1 0 0   1 1 0 0 0 0
        0 0 0 0 1 1   0 0 1 1 0 0   1 1 0 0 0 0
        """

        K, F, T = mask.shape

        assert K < 10, (K, 'Sure?')
        assert F % 2 == 1, (F, 'Sure? Usually F is odd.', mask.shape)

        # # Loopy code
        # mapping = np.zeros([K, F], np.int64)
        # mapping[:, 0] = range(K)
        #
        # for f in range(1, F):
        #     scores = self.get_score_matrix(
        #         mask[:, f, :],
        #         mask[:, f - 1, :],
        #         # axis=-1,
        #     )
        #     mapping[:, f] = _mapping_from_score_matrix(
        #         scores, algorithm='greedy')[mapping[:, f - 1]]

        scores = self.get_score_matrix(mask[:, 1:, :], mask[:, :-1, :])
        mapping = _mapping_from_score_matrix(scores, algorithm='greedy')
        # Append the mapping for the first frequency as identity
        mapping = np.append(
            np.arange(K, dtype=mapping.dtype)[:, None], mapping, axis=-1)

        # Recursively apply the mapping to the mapping
        for f in range(1, F):
            mapping[:, f] = mapping[mapping[:, f - 1], f]

        return mapping


class OraclePermutationAlignment(_PermutationAlignment):
    def __init__(self, similarity_metric='euclidean', algorithm='optimal'):
        assert algorithm in ['greedy', 'optimal'], algorithm

        self.get_score_matrix = getattr(_ScoreMatrix, similarity_metric)
        self.algorithm = algorithm

    def calculate_mapping(self, mask, reference_mask):
        """

        When only a global permutation problem should be solved, join the
        frequency axis and time frame axis.
        (i.e. `mask.reshape(*mask.shape[:-2], F*T)`)

        The time frame dimension is interpreted as vector dimension.
        The frequency dimension is interpreted as independent dimension.
        The sources dimension is interpreted as permuted dimension.

        Args:
            mask:
                shape: sources (K), frequencies (F), time frames (T)
            reference_mask:
                shape: source, frequency, time

        Returns:
            mapping:
                shape: sources (K), frequencies (F)

        >>> np.random.seed(0)
        >>> K, F, T = 3, 5, 6
        >>> reference_mask = np.zeros((K, F, T), dtype=np.int8)
        >>> def viz_mask(mask: np.ndarray):
        ...     mask = np.einsum('KFT->FKT', mask).astype(str).tolist()
        ...     for mask_f in mask:
        ...         print('   '.join([' '.join(m) for m in mask_f]))
        >>> viz_mask(reference_mask)
        0 0 0 0 0 0   0 0 0 0 0 0   0 0 0 0 0 0
        0 0 0 0 0 0   0 0 0 0 0 0   0 0 0 0 0 0
        0 0 0 0 0 0   0 0 0 0 0 0   0 0 0 0 0 0
        0 0 0 0 0 0   0 0 0 0 0 0   0 0 0 0 0 0
        0 0 0 0 0 0   0 0 0 0 0 0   0 0 0 0 0 0
        >>> reference_mask[0, :, :2] = 1
        >>> reference_mask[1, :, 2:4] = 1
        >>> reference_mask[2, :, 4:] = 1
        >>> viz_mask(reference_mask)
        1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
        1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
        1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
        1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
        1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
        >>> mask = apply_mapping(reference_mask, sample_random_mapping(K, F))
        >>> viz_mask(mask)
        0 0 0 0 1 1   0 0 1 1 0 0   1 1 0 0 0 0
        0 0 0 0 1 1   1 1 0 0 0 0   0 0 1 1 0 0
        1 1 0 0 0 0   0 0 0 0 1 1   0 0 1 1 0 0
        0 0 0 0 1 1   1 1 0 0 0 0   0 0 1 1 0 0
        0 0 0 0 1 1   0 0 1 1 0 0   1 1 0 0 0 0
        >>> viz_mask(OraclePermutationAlignment('cos')(mask, reference_mask))
        1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
        1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
        1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
        1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
        1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
        >>> viz_mask(OraclePermutationAlignment('euclidean')(mask, reference_mask))
        1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
        1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
        1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
        1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
        1 1 0 0 0 0   0 0 1 1 0 0   0 0 0 0 1 1
        """

        assert mask.shape == reference_mask.shape, (mask.shape, reference_mask.shape)

        K, *F, T = mask.shape

        assert K < 10, (K, 'Sure?')
        if len(F) == 1:
            assert F[0] % 2 == 1, (F, 'Sure? Usually F is odd.', mask.shape)

        score_matrix = self.get_score_matrix(mask, reference_mask)

        mapping = _mapping_from_score_matrix(score_matrix, self.algorithm)

        return mapping
