

class ComplexGaussianMixtureModel:
    """TV-cGMM.

    Higuchi, T.; Yoshioka, T. & Nakatani, T.
    Optimization of Speech Enhancement Front-End with Speech Recognition-Level
    Criterion
    Interspeech 2016, 2016, 3808-3812

    Original paper did not use mixture weights. In contrast to the original
    paper we make use of Eigen decomposition to avoid direct computation of
    determinant and inverse.

    This algorithm does not work well with too few channels.
    At least 4 channels are necessary for proper mask results.
    """

    def __init__(
            self, use_mixture_weights=False,
            eps=1e-10, visual_debug=False
    ):
        """Initializes empty instance variables.

        Shapes:
            pi: Mixture weights with shape (..., K).
            covariance: Covariance matrices with shape (..., K, D, D).
        """
        self.use_mixture_weights = use_mixture_weights
        self.eps = eps
        self.visual_debug = visual_debug
        self.covariance = np.empty((), dtype=np.complex)
        self.precision = np.empty((), dtype=np.complex)
        self.determinant = np.empty((), dtype=np.float)

        if self.use_mixture_weights:
            self.pi = np.empty((), dtype=np.float)

    def fit(
            self, Y, initialization, iterations=100,
            hermitize=True, trace_norm=True, eigenvalue_floor=1e-10,
            inverse='inv'
    ):
        """ EM for cGMM with any number of independent dimensions.

        Does not support sequence lengths.
        Can later be extended to accept more initializations.

        Args:
            Y: Mix with shape (..., T, D).
            iterations:
            initialization: Shape (..., K, T).
        """
        Y_for_psd = np.copy(np.swapaxes(Y, -2, -1), 'C')
        Y_for_pdf = np.copy(Y, 'C')
        D = Y_for_pdf.shape[-1]

        affiliations = np.copy(initialization)
        power = np.ones_like(affiliations)

        for i in range(iterations):
            # E step
            if i > 0:
                # Equation 10
                affiliations, power = self._predict(Y_for_pdf)

            # M step
            if self.use_mixture_weights:
                self.pi = np.mean(affiliations, axis=-1)

            # Equation 6
            self.covariance = get_power_spectral_density_matrix(
                Y_for_psd,
                np.copy(np.clip(affiliations, self.eps, 1 - self.eps) / power,
                        'C'),
                sensor_dim=-2, source_dim=-2, time_dim=-1
            )

            if hermitize:
                self.covariance = (
                                      self.covariance
                                      + np.swapaxes(self.covariance.conj(), -1,
                                                    -2)
                                  ) / 2

            if trace_norm:
                self.covariance /= np.einsum(
                    '...dd', self.covariance
                )[..., None, None]

            # Deconstructs covariance matrix and constrains eigenvalues
            eigenvals, eigenvecs = np.linalg.eigh(self.covariance)
            eigenvals = eigenvals.real
            eigenvals = np.maximum(
                eigenvals,
                np.max(eigenvals, axis=-1, keepdims=True) * eigenvalue_floor
            )
            diagonal = np.einsum('de,fkd->fkde', np.eye(D), eigenvals)
            self.covariance = np.einsum(
                'fkwx,fkxy,fkzy->fkwz', eigenvecs, diagonal, eigenvecs.conj()
            )
            self.determinant = np.prod(eigenvals, axis=-1)
            inverse_diagonal = np.einsum('de,fkd->fkde', np.eye(D),
                                         1 / eigenvals)
            self.precision = np.einsum(
                'fkwx,fkxy,fkzy->fkwz', eigenvecs, inverse_diagonal,
                eigenvecs.conj()
            )

            if self.visual_debug:
                with context_manager(figure_size=(24, 3)):
                    plt.plot(np.log10(
                        np.max(eigenvals, axis=-1)
                        / np.min(eigenvals, axis=-1)
                    ))
                    plt.xlabel('frequency bin')
                    plt.ylabel('eigenvalue spread')
                    plt.show()
                with context_manager(figure_size=(24, 3)):
                    plt.plot(self.pi)
                    plt.show()

    def _predict(self, Y, inverse='inv'):
        D = Y.shape[-1]

        if inverse == 'inv':
            precision = np.linalg.inv(self.covariance)
            power = np.abs(
                np.einsum('...td,...kde,...te->...kt', Y.conj(), precision, Y)
            ) / D + self.eps
        elif inverse == 'solve':
            Y_with_inverse_covariance = np.linalg.solve(
                self.covariance[..., None, :, :],
                Y[..., None, :, :]
            )
            power = np.einsum(
                '...td,...ktd->...kt', Y.conj(), Y_with_inverse_covariance
            ).real / D + self.eps
        elif inverse == 'eig':
            power = np.abs(
                np.einsum('...td,...kde,...te->...kt', Y.conj(), self.precision,
                          Y)
            ) / D + self.eps

        affiliations = np.exp(
            -np.log(self.determinant)[..., None] - D * np.log(power))

        if self.use_mixture_weights:
            affiliations *= self.pi[..., None]

        affiliations /= np.sum(affiliations, axis=-2, keepdims=True) + self.eps

        if self.visual_debug:
            _plot_affiliations(affiliations)

        return affiliations, power

    def predict(self, Y, inverse='inv'):
        """Predict class affiliation posteriors from given model.

        Args:
            Y: Mix with shape (..., T, D).
        Returns: Affiliations with shape (..., K, T).
        """
        return self._predict(Y, inverse=inverse)[0]