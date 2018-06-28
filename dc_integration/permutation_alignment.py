import numpy as np


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
            raise StopIteration()


class DHTVPermutationAlignment:
    """Encapsulates all frequency permutation related functions.

    The main algorithm is an implementation of the unpublished frequency
    permutation alignment algorithm from [TranVu2015BSS].

    This does not solve the global permutation problem.
    """
    def __init__(
            self, *,
            stft_size,
            segment_start, segment_width, segment_shift,
            main_iterations, sub_iterations
    ):
        self.stft_size = stft_size
        self.segment_start = segment_start
        self.segment_width = segment_width
        self.segment_shift = segment_shift
        self.main_iterations = main_iterations
        self.sub_iterations = sub_iterations

    @classmethod
    def from_stft_size(cls, stft_size):
        """ Use this, if you do not want to set all parameters manually."""
        if stft_size == 512:
            return cls(
                stft_size=stft_size,
                segment_start=70, segment_width=100, segment_shift=20,
                main_iterations=20, sub_iterations=2
            )
        elif stft_size == 1024:
            return cls(
                stft_size=stft_size,
                segment_start=100, segment_width=100, segment_shift=20,
                main_iterations=20, sub_iterations=2
            )
        else:
            raise ValueError(
                'There is no default for stft_size={}.', stft_size
            )

    @staticmethod
    def _align_segment(mask, prototype):
        """Calculates permutation for a single frequency.

        An example can be found in [Boeddecker2015Free] on page 33/ 34.

        Args:
            prototype: Often called prototype or centroid with shape (K, T).
            mask: The permuted mask for the given frequency, shape (K, T).
        Returns:
            Reverse permutation.
        """
        K = prototype.shape[0]
        c_matrix = np.dot(prototype, mask.T)

        reverse_permutation = np.zeros((K,), dtype=np.int)
        estimated_permutation = np.zeros((K,), dtype=np.int)

        for _ in range(K):
            c_max = np.max(c_matrix, axis=0)
            index_0 = np.argmax(c_matrix, axis=0)
            index_1 = np.argmax(c_max)
            c_matrix[index_0[index_1], :] = -1
            c_matrix[:, index_1] = -1
            reverse_permutation[index_0[index_1]] = index_1
            estimated_permutation[index_1] = index_0[index_1]

        return reverse_permutation

    @staticmethod
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

    @property
    def alignment_plan(self):
        """Provides the alignment plan for a given setup. Nice to plot, too.

        >>> import matplotlib.pyplot as plt
        >>> alignment_plan \
        ...     = DHTVPermutationAlignment.from_stft_size(512).alignment_plan
        >>> fig, ax = plt.subplots()
        >>> for i, s in enumerate(alignment_plan):
        ...     _ = ax.broken_barh([[s[1], s[2] - s[1]]] , (i - 0.25, 0.5))
        >>> _ = ax.set_xlabel('Frequency bin')
        >>> _ = ax.set_ylabel('Iteration')
        >>> plt.show()
        """
        F = self.stft_size // 2 + 1
        alignment_plan_lower_start = range(
            self.segment_start + self.segment_shift,
            F - self.segment_width, self.segment_shift
        )

        alignment_plan_higher_start = range(
            self.segment_start - self.segment_shift, 0, -self.segment_shift
        )

        alignment_plan_start = interleave(
            alignment_plan_lower_start, alignment_plan_higher_start
        )

        alignment_plan = [
            [
                self.main_iterations, self.segment_start,
                self.segment_start + self.segment_width
            ]
        ] + [
            [
                self.sub_iterations, s, s + self.segment_width
            ] for s in alignment_plan_start
        ]

        alignment_plan[2 * len(alignment_plan_higher_start)][1] = 0
        alignment_plan[-1][2] = F
        return alignment_plan

    def calculate_mapping(self, mask):
        """Returns just the mapping based on permuted mask input.

        Args:
            mask: Permuted mask with shape (K, F, T).
        Returns:
            Reverse mapping with shape (K, F).
        """
        K, F, _ = mask.shape

        # (K, F, T)
        features = mask / np.linalg.norm(mask, axis=-1, keepdims=True)

        mapping = np.repeat(np.arange(K)[:, None], F, axis=1)

        for iterations, start, end in self.alignment_plan:
            for _ in range(iterations):
                # (K, T)
                centroid = np.sum(features[:, start:end, :], axis=1)
                centroid /= np.linalg.norm(centroid, axis=-1, keepdims=True)

                break_flag = False
                for f in range(start, end):
                    reverse_permutation = self._align_segment(
                        features[:, f, :], centroid,
                    )
                    if not (reverse_permutation == list(range(K))).all():
                        break_flag = True
                        features[:, f, :] = features[reverse_permutation, f, :]
                        mapping[:, f] = mapping[reverse_permutation, f]
                if break_flag:
                    break

        return mapping

    @staticmethod
    def apply_mapping(mask, mapping):
        """Applies the mapping to obtain a frequency aligned mask.

        Args:
            mask: Permuted mask with shape (K, F, T).
            mapping: Reverse mapping with shape (K, F).
        """
        _, F = mapping.shape
        aligned_mask = np.zeros_like(mask)
        for f in range(F):
            aligned_mask[:, f, :] = mask[mapping[:, f], f, :]
        return aligned_mask

    def __call__(self, mask):
        """Calculates mapping and applies it to the provided mask.

        Args:
            mask: Permuted mask with shape (K, F, T).
        """
        mapping = self.calculate_mapping(mask)
        return self.apply_mapping(mask, mapping)
