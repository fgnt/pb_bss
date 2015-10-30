import unittest
from os import path

import numpy as np

import nt.testing as tc
from nt.speech_enhancement.mask_estimation import estimate_IBM as ideal_binary_mask
from nt.speech_enhancement.beamformer import get_power_spectral_density_matrix
from nt.speech_enhancement.beamformer import get_pca_vector
from nt.speech_enhancement.beamformer import get_mvdr_vector
from nt.speech_enhancement.beamformer import get_gev_vector
from nt.speech_enhancement.beamformer import get_lcmv_vector
from nt.utils.math_ops import vector_H_vector
from nt.utils.matlab import Mlab, matlab_test
from nt.speech_enhancement.mask_estimation import simple_ideal_soft_mask


# uncomment, if you want to test matlab functions
# matlab_test = unittest.skipUnless(True,'matlab-test')

def rand(*shape, data_type):
    if not shape:
        shape = (1,)
    elif isinstance(shape[0], tuple):
        shape = shape[0]

    def uniform(data_type_local):
        return np.random.uniform(-1, 1, shape).astype(data_type_local)

    if data_type in (np.float32, np.float64):
        return uniform(data_type)
    elif data_type is np.complex64:
        return uniform(np.float32) + 1j * uniform(np.float32)
    elif data_type is np.complex128:
        return uniform(np.float64) + 1j * uniform(np.float64)


class TestPSDMatrix(unittest.TestCase):
    # or covariance matrix

    F, T, D, K = 51, 31, 6, 2

    def generate_date(self, x_shape, mask_shape):
        x = rand(x_shape, data_type=np.complex128)
        mask = np.random.uniform(0, 1, mask_shape)
        mask = mask / np.sum(mask, axis=0, keepdims=True)
        return x, mask

    def gererate_and_verify_psd(self, x_shape, mask_shape, psd_shape=None):
        x, mask = self.generate_date(x_shape, mask_shape)
        if mask_shape is None:
            psd = get_power_spectral_density_matrix(x)
        else:
            psd = get_power_spectral_density_matrix(x, mask)
        if psd_shape is not None:
            tc.assert_equal(psd.shape, psd_shape)
        tc.assert_hermitian(psd)
        tc.assert_positive_semidefinite(psd)

    def test_PSD_without_mask(self):
        self.gererate_and_verify_psd((self.D, self.T), None, psd_shape=(self.D, self.D))

    def test_PSD_with_mask(self):
        self.gererate_and_verify_psd((self.D, self.T), (self.T,), psd_shape=(self.D, self.D))

    def test_PSD_with_mask_with_source(self):
        self.gererate_and_verify_psd((self.D, self.T), (self.K, self.T,), psd_shape=(self.K, self.D, self.D))

    def test_PSD_with_mask_independent_dim(self):
        self.gererate_and_verify_psd((self.F, self.D, self.T), (self.F, self.T,), psd_shape=(self.F, self.D, self.D))
        self.gererate_and_verify_psd((self.F, self.F, self.D, self.T), (self.F, self.F, self.T,),
                                     psd_shape=(self.F, self.F, self.D, self.D))

    def test_PSD_with_mask_independent_dim_with_source(self):
        self.gererate_and_verify_psd((self.F, self.D, self.T), (self.F, self.K, self.T,),
                                     psd_shape=(self.F, self.K, self.D, self.D))
        self.gererate_and_verify_psd((self.F, self.F, self.D, self.T), (self.F, self.F, self.K, self.T,),
                                     psd_shape=(self.F, self.F, self.K, self.D, self.D))

    def test_PSD_without_mask_independent_dim(self):
        self.gererate_and_verify_psd((self.F, self.D, self.T), None, psd_shape=(self.F, self.D, self.D))
        self.gererate_and_verify_psd((self.F, self.F, self.D, self.T), (self.F, self.F, self.T,),
                                     psd_shape=(self.F, self.F, self.D, self.D))

    def test_predict_output(self):
        x, _ = self.generate_date((self.D,), None)
        x_rep = x[:, np.newaxis].repeat((self.T,), 1)
        psd = get_power_spectral_density_matrix(x_rep)

        psd_predict = x[:, np.newaxis].dot(x[np.newaxis, :].conj())
        tc.assert_almost_equal(psd, psd_predict)

    def test_predict_output_with_mask(self):
        x, _ = self.generate_date((self.D,), None)
        x_rep = x[:, np.newaxis].repeat((self.T,), 1)
        mask = np.ones((self.T,))
        psd = get_power_spectral_density_matrix(x_rep, mask)
        psd2 = get_power_spectral_density_matrix(x_rep, mask * 2)
        psd3 = get_power_spectral_density_matrix(x_rep, mask * 0.5)

        psd_predict = x[:, np.newaxis].dot(x[np.newaxis, :].conj())
        tc.assert_almost_equal(psd, psd_predict)
        tc.assert_almost_equal(psd, psd2)
        tc.assert_almost_equal(psd, psd3)

    def test_different_valued_masks_output(self):
        x, mask = self.generate_date((self.F, self.D, self.T), (self.F, self.T,))

        psd = get_power_spectral_density_matrix(x, mask)
        psd2 = get_power_spectral_density_matrix(x, mask*2)
        psd3 = get_power_spectral_density_matrix(x, mask*0.5)

        tc.assert_almost_equal(psd, psd2)
        tc.assert_almost_equal(psd, psd3)


class TestBeamformerMethods(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        datafile = path.join(path.dirname(path.realpath(__file__)), 'data.npz')
        datafile_multi_speaker = path.join(path.dirname(path.realpath(__file__)), 'data_multi_speaker.npz')

        self.mlab = Mlab()

        if not path.exists(datafile_multi_speaker):
            self.generate_source_file_with_matlab(mlab=self.mlab)

        with np.load(datafile) as data:
            X = data['X']
            Y = data['Y']
            N = data['N']
        ibm = ideal_binary_mask(X[:, 4, :], N[:, 4, :])
        self.Y_bf, self.X_bf, self.N_bf = Y.T, X.T, N.T
        self.ibm_X_bf = ibm[0].T
        self.ibm_N_bf = ibm[1].T
        self.ibm_X_bf_th = np.maximum(self.ibm_X_bf, 1e-4)
        self.ibm_N_bf_th = np.maximum(self.ibm_N_bf, 1e-4)
        self.Phi_XX = get_power_spectral_density_matrix(self.Y_bf, self.ibm_X_bf_th)
        self.Phi_NN = get_power_spectral_density_matrix(self.Y_bf, self.ibm_N_bf_th)
        self.Phi_NN = self.Phi_NN + np.tile(1e-10 * np.eye(self.Phi_NN.shape[1]), (self.Phi_NN.shape[0], 1, 1))
        self.W_pca = get_pca_vector(self.Phi_XX)
        self.W_mvdr = get_mvdr_vector(self.W_pca, self.Phi_NN)
        self.W_gev = get_gev_vector(self.Phi_XX, self.Phi_NN)

        with np.load(datafile_multi_speaker) as data:
            X = data['X']  # K F D T
            Y = data['Y']  # F D T
            N = data['N']  # F D T
            self.data_multi_speaker = {'X': data['X'], 'Y': data['Y'], 'N': data['N']}

        X_mask, N_mask = simple_ideal_soft_mask(X, N, source_dim=0, feature_dim=-2, tuple_output=True)

        Phi_XX = get_power_spectral_density_matrix(Y, X_mask, source_dim=0)  # K F D D
        Phi_NN = get_power_spectral_density_matrix(Y, N_mask)  # F D D

        W_pca = get_pca_vector(Phi_XX)
        W_mvdr = get_mvdr_vector(W_pca, Phi_NN)
        W_gev = np.zeros_like(W_mvdr)
        print(Phi_XX.shape)
        print(W_gev.shape)
        W_gev[0, :, :] = get_gev_vector(Phi_XX[0, :, :, :], Phi_NN)
        W_gev[1, :, :] = get_gev_vector(Phi_XX[1, :, :, :], Phi_NN)

        W_lcmv = get_lcmv_vector(W_pca, [1, 0], Phi_NN)

    @staticmethod
    def generate_source_file_with_matlab(mlab=Mlab()):
        import nt.transform as transform

        # ToDo: replace with Python funktions (current missing)
        mlab.run_code('D = 6;')  # Number of microphones
        mlab.run_code('K = 2;')  # Number of speakers
        mlab.run_code('tSignal = 5;')  # Signal length in seconds
        mlab.run_code('seed = 1;')
        mlab.run_code('SNR = 15;')
        mlab.run_code('soundDecayTime = 0.16;')
        mlab.run_code('samplingRate = 16000;')
        mlab.run_code("noiseType = 'whiteGaussian';")
        mlab.run_code('rirFilterLength = 2^14;')
        mlab.run_code('fftSize = 2^10;')
        mlab.run_code('fftShiftFactor = 1/4;')
        mlab.run_code('analysisWindowHandle = @(x) blackman(x);')

        mlab.run_code('sourceDoA =  degtorad([30, -30, 60, -60, 90, -90, 15, -15, 45, -45, 75, -75, 0]);')
        mlab.run_code('sourceDoA = sourceDoA(1:K);')
        mlab.run_code('rir = zeros(rirFilterLength, 8, numel(sourceDoA));')
        mlab.run_code('for ii = 1 : numel(sourceDoA)'
                      '    [rir(:, :, ii), sensorPositions] = reverb.acquireMIRD(1, 1, sourceDoA(ii), samplingRate, rirFilterLength, soundDecayTime);'
                      'end')
        mlab.run_code('sensorsID = [4 5 3 6 2 7 1 8];')
        mlab.run_code('sensorsID = sensorsID(1:D);')
        mlab.run_code('sensorsID = sort(sensorsID);')
        mlab.run_code('sensorPositions = sensorPositions(:, sensorsID);')
        mlab.run_code('rir = rir(:, sensorsID, :);')

        mlab.run_code('speakers = database.acquireSignals(tSignal, samplingRate,  K, seed);')
        mlab.run_code(
            "noise = database.noise(tSignal, samplingRate, D, seed, 'noiseType', noiseType, 'sensorPositions', sensorPositions);")

        mlab.run_code(
            "[~, speakers, noise] = bss.generate.convolutiveMixture(speakers, rir, noise, SNR, 'sourceScaled');")

        speakers = mlab.get_variable('speakers')
        noise = mlab.get_variable('noise')

        Speakers = transform.stft(speakers)
        Noise = transform.stft(noise)

        Y = np.sum(np.concatenate((Speakers, Noise[:, :, :, np.newaxis]), axis=3), axis=3).transpose(1, 2, 0).copy()
        X = Speakers.transpose(3, 1, 2, 0).copy()
        N = Noise.transpose(1, 2, 0).copy()

        datafile_multi_speaker = path.join(path.dirname(path.realpath(__file__)), 'data_multi_speaker.npz')
        np.savez(datafile_multi_speaker, X=X, Y=Y, N=N)

    @matlab_test
    def test_compare_PSD_without_mask(self):
        mlab = self.mlab.process
        mlab.set_variable('Y', self.Y_bf)
        mlab.run_code('Phi = random.covariance(Y, [], 3, 2);')
        Phi_matlab = mlab.get_variable('Phi')
        Phi = get_power_spectral_density_matrix(self.Y_bf)
        tc.assert_allclose(Phi, Phi_matlab, atol=1e-4)

    @matlab_test
    def test_compare_PSD_with_mask(self):
        mlab = self.mlab.process
        mlab.set_variable('Y', self.Y_bf)
        mlab.set_variable('ibm', self.ibm_N_bf[:, np.newaxis, :].astype(np.float))
        mlab.run_code('Phi = random.covariance(Y, ibm, 3, 2);')
        Phi_matlab = mlab.get_variable('Phi')
        Phi = get_power_spectral_density_matrix(self.Y_bf, self.ibm_N_bf)
        tc.assert_allclose(Phi, Phi_matlab, atol=1e-4)

    @matlab_test
    def test_compare_PCA_beamformer(self):
        mlab = self.mlab.process
        mlab.set_variable('Phi_XX', self.Phi_XX)
        mlab.run_code("W = bss.beamformer.pca('cleanObservationMatrix', Phi_XX);")
        W_matlab = mlab.get_variable('W')
        tc.assert_cosine_similarity(W_matlab, self.W_pca)

    def test_mvdr_beamformer(self):
        tc.assert_allclose(vector_H_vector(self.W_pca, self.W_mvdr), 1)

    @matlab_test
    def test_compare_mvdr_beamformer(self):
        mlab = self.mlab.process
        mlab.set_variable('Phi_NN', self.Phi_NN)
        mlab.set_variable('lookDirection', self.W_pca)
        mlab.run_code("W = bss.beamformer.mvdr('noiseMatrix', Phi_NN, 'lookDirection', lookDirection);")
        W_matlab = mlab.get_variable('W')
        tc.assert_cosine_similarity(W_matlab, self.W_mvdr)

    @matlab_test
    def test_compare_gev_beamformer(self):
        mlab = self.mlab.process
        mlab.set_variable('Phi_XX', self.Phi_XX)
        mlab.set_variable('Phi_NN', self.Phi_NN)
        mlab.run_code("W = bss.beamformer.gev('cleanObservationMatrix', Phi_XX, 'noiseMatrix', Phi_NN);")
        W_matlab = mlab.get_variable('W')
        tc.assert_cosine_similarity(W_matlab, self.W_gev)

    @matlab_test
    def test_compare_lcmv_beamformer(self):
        from nt.speech_enhancement.beamformer import apply_beamforming_vector
        from nt.transform.module_stft import istft_loop
        from nt.evaluation import sxr

        data = self.data_multi_speaker
        X = data['X']  # K F D T
        Y = data['Y']  # F D T
        N = data['N']  # F D T

        # Caculate masks
        # X_mask.shape = (2, 513, 316)
        # N_mask.shape = (513, 316)
        X_mask, N_mask = simple_ideal_soft_mask(X, N, source_dim=0, feature_dim=-2, tuple_output=True)

        # Phi_XX.shape = (2, 513, 6, 6)
        # Phi_NN.shape = (513, 6, 6)
        Phi_XX = get_power_spectral_density_matrix(Y, X_mask, source_dim=0)  # K F D D
        Phi_NN = get_power_spectral_density_matrix(Y, N_mask)  # F D D

        # W_pca.shape = (2, 513, 6)
        W_pca = get_pca_vector(Phi_XX)

        # W_lcmv1.shape = (513, 6)
        # W_lcmv2.shape = (513, 6)
        # W_lcmv.shape = (2, 513, 6)
        W_lcmv1 = get_lcmv_vector(W_pca, [1, 0], Phi_NN)
        W_lcmv2 = get_lcmv_vector(W_pca, [0, 1], Phi_NN)
        W_lcmv = np.array(([W_lcmv1, W_lcmv2]))

        # W_pca_tmp.shape = (513, 2, 6)
        W_pca_tmp = W_pca.transpose(1, 2, 0)
        Phi_NN_tmp = Phi_NN

        mlab = self.mlab
        mlab.set_variable('W_pca', W_pca_tmp)
        mlab.set_variable('Phi_NN', Phi_NN_tmp)
        mlab.run_code_print('size(W_pca)')
        mlab.run_code_print('size(Phi_NN)')
        mlab.run_code(
            "[W(:, :, 1), failing] = bss.beamformer.lcmv('observationMatrix', Phi_NN, 'lookDirection', W_pca, 'responseVector', [1 0]);")
        mlab.run_code(
            "W(:, :, 2) = bss.beamformer.lcmv('observationMatrix', Phi_NN, 'lookDirection', W_pca, 'responseVector', [0 1]);")
        W_matlab = mlab.get_variable('W')
        failing = mlab.get_variable('failing')
        print(np.sum(failing))
        assert (np.sum(failing) == 0.0)

        def sxr_output(W):
            Shat = np.zeros((2, 2, 513, 316), dtype=complex)
            Shat[0, 0, :, :] = apply_beamforming_vector(W[0, :, :], X[0, :, :, :])
            Shat[1, 1, :, :] = apply_beamforming_vector(W[1, :, :], X[1, :, :, :])
            Shat[0, 1, :, :] = apply_beamforming_vector(W[1, :, :], X[0, :, :, :])
            Shat[1, 0, :, :] = apply_beamforming_vector(W[0, :, :], X[1, :, :, :])

            Nhat = np.zeros((2, 513, 316), dtype=complex)
            Nhat[0, :, :] = apply_beamforming_vector(W[0, :, :], N)
            Nhat[0, :, :] = apply_beamforming_vector(W[0, :, :], N)
            shat = istft_loop(Shat, time_dim=-1, freq_dim=-2)
            nhat = istft_loop(Nhat, time_dim=-1, freq_dim=-2)
            return sxr.output_sxr(shat.transpose(2, 0, 1), nhat.transpose())

        W_matlab_tmp = W_matlab.transpose(2, 0, 1)
        W_lcmv_tmp = W_lcmv

        sxr_matlab = sxr_output(W_matlab_tmp)
        sxr_py = sxr_output(W_lcmv_tmp)

        tc.assert_almost_equal(sxr_matlab, sxr_py)

        tc.assert_cosine_similarity(W_matlab_tmp, W_lcmv_tmp)
