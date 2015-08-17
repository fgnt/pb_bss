import unittest
import nt.testing as tc
import numpy as np
from nt.speech_enhancement.mask_estimation import estimate_IBM as ideal_binary_mask
from nt.speech_enhancement.beamformer import get_power_spectral_density_matrix
from pymatbridge import Matlab
from nt.speech_enhancement.beamformer import get_pca_vector
from nt.speech_enhancement.beamformer import get_mvdr_vector
from nt.speech_enhancement.beamformer import get_gev_vector
from nt.speech_enhancement.beamformer import normalize_vector_to_unit_length

from os import environ
matlab = unittest.skipUnless(environ.get('TEST_MATLAB'),'matlab-test')

class TestBeamformerMethods(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        with np.load('data.npz') as data:
            X = data['X']
            Y= data['Y']
            N=data['N']
        ibm = ideal_binary_mask(X[:, 4, :], N[:, 4, :])
        self.Y_bf, self.X_bf, self.N_bf = Y.T, X.T, N.T
        self.ibm_X_bf = ibm[0].T
        self.ibm_N_bf = ibm[1].T
        self.ibm_X_bf_th = np.maximum(self.ibm_X_bf, 1e-4)
        self.ibm_N_bf_th = np.maximum(self.ibm_N_bf, 1e-4)
        self.Phi_XX = get_power_spectral_density_matrix(self.Y_bf, self.ibm_X_bf_th)
        self.Phi_NN = get_power_spectral_density_matrix(self.Y_bf, self.ibm_N_bf_th)
        self.W_pca = get_pca_vector(self.Phi_XX)
        self.W_mvdr = get_mvdr_vector(self.W_pca, self.Phi_NN)
        self.W_gev = get_gev_vector(self.Phi_XX, self.Phi_NN)
        if environ.get('TEST_MATLAB'):
            self.mlab = Matlab('nice -n 3 /net/ssd/software/MATLAB/R2015a/bin/matlab -nodisplay -nosplash')
            self.mlab.start()
            _ = self.mlab.run_code('run /net/home/ldrude/Projects/2015_python_matlab/matlab/startup.m')

    @matlab
    def test_compare_PSD_without_mask(self):
        self.mlab.set_variable('Y', self.Y_bf)
        self.mlab.run_code('Phi = random.covariance(Y, [], 3, 2);')
        Phi_matlab = self.mlab.get_variable('Phi')
        Phi = get_power_spectral_density_matrix(self.Y_bf)
        tc.assert_allclose(Phi, Phi_matlab, atol=1e-4)

    @matlab
    def test_compare_PSD_with_mask(self):
        self.mlab.set_variable('Y', self.Y_bf)
        self.mlab.set_variable('ibm', self.ibm_N_bf[:, np.newaxis, :].astype(np.float))
        self.mlab.run_code('Phi = random.covariance(Y, ibm, 3, 2);')
        Phi_matlab = self.mlab.get_variable('Phi')
        Phi = get_power_spectral_density_matrix(self.Y_bf, self.ibm_N_bf)
        tc.assert_allclose(Phi, Phi_matlab, atol=1e-4)

    @matlab
    def test_compare_PCA_beamformer(self):
        self.mlab.set_variable('Phi_XX', self.Phi_XX)
        self.mlab.run_code("W = bss.beamformer.pca('cleanObservationMatrix', Phi_XX);")
        W_matlab = self.mlab.get_variable('W')
        distance = 1 - np.abs(self.vector_H_vector(self.W_pca, W_matlab))**2
        tc.assert_array_less(distance, 1e-6)

    def test_mvdr_beamformer(self):
        tc.assert_allclose(self.vector_H_vector(self.W_pca, self.W_mvdr), 1)

    @matlab
    def test_compare_mvdr_beamformer(self):
        # missing comparison to matlab result
        tc.assert_array_equal(0, 1)

    @matlab
    def test_compar_gev_beamformer(self):
        self.mlab.set_variable('Phi_XX', self.Phi_XX)
        self.mlab.set_variable('Phi_NN', self.Phi_NN)
        self.mlab.run_code("W = bss.beamformer.gev('cleanObservationMatrix', Phi_XX, 'noiseMatrix', Phi_NN);")
        W_matlab = self.mlab.get_variable('W')
        distance = 1 - np.abs(self.vector_H_vector(self.W_gev, W_matlab))**2
        tc.assert_array_less(distance, 1e-6)

    def test_Unit_length_normalization(self):
        W_normalized = normalize_vector_to_unit_length(self.W_gev)
        tc.assert_allclose(self.vector_H_vector(W_normalized, W_normalized), 1)

    @staticmethod
    def vector_H_vector(x, y):
        return np.einsum('...a,...a->...', x.conj(), y)
