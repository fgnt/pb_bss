import unittest
import nt.testing as tc
import numpy as np
from nt.speech_enhancement.mask_estimation import estimate_IBM as ideal_binary_mask
from nt.speech_enhancement.beamformer import get_power_spectral_density_matrix
from pymatbridge import Matlab
from nt.speech_enhancement.beamformer import get_pca_vector
from nt.speech_enhancement.beamformer import get_mvdr_vector
from nt.speech_enhancement.beamformer import get_gev_vector
from os import path
from os import environ
from cached_property import cached_property
from nt.utils import math

class Mlab():
    @cached_property
    def process(self):
        mlab_process = Matlab('nice -n 3 /net/ssd/software/MATLAB/R2015a/bin/matlab -nodisplay -nosplash')
        mlab_process.start()
        _ = mlab_process.run_code('run /net/home/ldrude/Projects/2015_python_matlab/matlab/startup.m')
        return mlab_process

# define decorator to skip matlab_tests
matlab = unittest.skipUnless(environ.get('TEST_MATLAB'),'matlab-test')

class TestBeamformerMethods(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        datafile = path.join(path.dirname(path.realpath(__file__)), 'data.npz')
        with np.load(datafile) as data:
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
        self.mlab = Mlab()

    #@matlab
    def test_compare_PSD_without_mask(self):
        mlab = self.mlab.process
        mlab.set_variable('Y', self.Y_bf)
        mlab.run_code('Phi = random.covariance(Y, [], 3, 2);')
        Phi_matlab = mlab.get_variable('Phi')
        Phi = get_power_spectral_density_matrix(self.Y_bf)
        tc.assert_allclose(Phi, Phi_matlab, atol=1e-4)

    #@matlab
    def test_compare_PSD_with_mask(self):
        mlab = self.mlab.process
        mlab.set_variable('Y', self.Y_bf)
        mlab.set_variable('ibm', self.ibm_N_bf[:, np.newaxis, :].astype(np.float))
        mlab.run_code('Phi = random.covariance(Y, ibm, 3, 2);')
        Phi_matlab = mlab.get_variable('Phi')
        Phi = get_power_spectral_density_matrix(self.Y_bf, self.ibm_N_bf)
        tc.assert_allclose(Phi, Phi_matlab, atol=1e-4)

    #@matlab
    def test_compare_PCA_beamformer(self):
        mlab = self.mlab.process
        mlab.set_variable('Phi_XX', self.Phi_XX)
        mlab.run_code("W = bss.beamformer.pca('cleanObservationMatrix', Phi_XX);")
        W_matlab = mlab.get_variable('W')
        distance = 1 - np.abs(math.vector_H_vector(self.W_pca, W_matlab))**2
        tc.assert_array_less(distance, 1e-6)

    def test_mvdr_beamformer(self):
        tc.assert_allclose(math.vector_H_vector(self.W_pca, self.W_mvdr), 1)

    #@matlab
    def test_compare_mvdr_beamformer(self):
        mlab = self.mlab.process
        mlab.set_variable('Phi_NN', self.Phi_NN)
        mlab.set_variable('lookDirection', self.W_pca)
        mlab.run_code("W = bss.beamformer.mvdr('noiseMatrix', Phi_NN, 'lookDirection', lookDirection);")
        W_matlab = mlab.get_variable('W')
        tc.assert_cosine_similarity(W_matlab, self.W_mvdr)

    #@matlab
    def test_compar_gev_beamformer(self):
        mlab = self.mlab.process
        mlab.set_variable('Phi_XX', self.Phi_XX)
        mlab.set_variable('Phi_NN', self.Phi_NN)
        mlab.run_code("W = bss.beamformer.gev('cleanObservationMatrix', Phi_XX, 'noiseMatrix', Phi_NN);")
        W_matlab = mlab.get_variable('W')
        distance = 1 - np.abs(math.vector_H_vector(self.W_gev, W_matlab))**2
        tc.assert_array_less(distance, 1e-6)
