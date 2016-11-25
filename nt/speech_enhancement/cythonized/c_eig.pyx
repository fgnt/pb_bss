# encoding: utf-8

cimport numpy as np
import numpy as np
from scipy.linalg.cython_lapack cimport zggev
cimport cython

#http://stackoverflow.com/questions/18593308/tips-for-optimising-code-in-cython
# importing math functions from a C-library (faster than numpy)
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI

@cython.boundscheck(False)
@cython.wraparound(False)
def _cythonized_eig(np.ndarray[complex, ndim=3] a, np.ndarray[complex, ndim=3] b):
    """
    Returns the GEV beamforming vector.

    :param target_psd_matrix: Target PSD matrix
        with shape (bins, sensors, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (bins, sensors)
    """

    a = np.asfortranarray(a.transpose(1, 2, 0))
    b = np.asfortranarray(b.transpose(1, 2, 0))

    # Get the dimensions
    cdef size_t sensors = a.shape[0]
    cdef size_t bins = a.shape[2]
    cdef int N = a.shape[0]
    cdef int LDA = a.shape[1]
    cdef int LDB = b.shape[1]
    cdef int LDVL = sensors
    cdef int LDVR = sensors

    # cdef np.ndarray[complex, ndim=3] a = a[:]
    # cdef np.ndarray[complex, ndim=3] b = b[:]
    cdef np.ndarray[complex, ndim=1] alpha = np.empty(sensors, dtype=np.complex)
    cdef np.ndarray[complex, ndim=1] beta = np.empty(sensors, dtype=np.complex)
    cdef np.ndarray[complex, ndim=1] work_tmp = np.empty(2, dtype=np.complex)
    # Not referenced since JOBVL == N
    cdef np.ndarray[complex, ndim=3] vl = np.empty((bins, sensors, sensors),
                                                   dtype=np.complex)
    cdef np.ndarray[complex, ndim=3] vr = np.empty((bins, sensors, sensors),
                                                   dtype=np.complex)

    cdef complex* alpha_ptr = <complex*> alpha.data
    cdef complex* beta_ptr = <complex*> beta.data
    cdef complex* vl_ptr = <complex*> vl.data
    cdef complex* vr_ptr = <complex*> vr.data
    cdef complex[:, :, :] a_view = a
    cdef complex[:, :, :] b_view = b
    cdef complex* work_tmp_ptr = <complex*> work_tmp.data

    cdef size_t f
    cdef char JOBVL = 'N'
    cdef char JOBVR = 'V'
    cdef int tmp_size = -1
    cdef int INFO = 0

    # Get the size of the workspace
    cdef complex* a_ptr = &a_view[0, 0, 0]
    cdef complex* b_ptr = &b_view[0, 0, 0]

    cdef np.ndarray[double, ndim=1] rwork = np.empty(8*sensors, dtype=np.float64)
    cdef double* rwork_ptr = <double*> rwork.data

    zggev(&JOBVL, &JOBVR, &N, a_ptr, &LDA, b_ptr, &LDB, alpha_ptr, beta_ptr,
           vl_ptr, &LDVL, vr_ptr, &LDVR, work_tmp_ptr, &tmp_size, rwork_ptr, &INFO)

    cdef int work_size = <int> abs(work_tmp[0])

    # Create workspace
    cdef np.ndarray[complex, ndim=1] work = np.empty(work_size,
                                                     dtype=np.complex, order='F')

    work_ptr = <complex*> work.data
    rwork_ptr = <double*> rwork.data

    cdef np.ndarray[int, ndim=1] max_idx = np.empty(bins, dtype=np.int32)

    cdef int[:] max_idx_view = max_idx
    cdef int* max_idx_ptr = <int*> max_idx.data
    cdef size_t w_idx
    cdef double w_max = 0

    cdef complex* a_ptr_global = &a_view[0, 0, 0]
    cdef complex* b_ptr_global = &b_view[0, 0, 0]
    cdef complex* vr_ptr_global = <complex*> vr.data

    cdef int sensors_square = sensors * sensors

    cdef np.ndarray[complex, ndim=2] w = np.empty((bins, sensors),
                                                      dtype=np.complex)

    cdef size_t m
    for f in range(bins):
        a_ptr = a_ptr_global + sensors_square * f
        b_ptr = b_ptr_global + sensors_square * f
        vr_ptr = vr_ptr_global + sensors_square * f
        zggev(&JOBVL, &JOBVR, &N, a_ptr, &LDA, b_ptr, &LDB, alpha_ptr, beta_ptr,
           vl_ptr, &LDVL, vr_ptr, &LDVR, work_ptr, &work_size, rwork_ptr, &INFO)
        if INFO != 0:
            if INFO < 0:
                raise ValueError('Value {} has an illegal value for '
                                 'frequency {}'.format(-INFO, f))
            elif 0 < INFO <= sensors:
                raise ValueError('''The QZ iteration failed.  No eigenvectors have been
                calculated, but ALPHA(j) and BETA(j) should be
                correct for j={}+1,...,N.'''.format(
                    INFO))
            elif INFO == sensors + 1:
                raise ValueError('''other then QZ iteration failed in DHGEQ''')
            else:
                raise ValueError('error return from DTGEVC.')

        for m in range(sensors):
            w[f][m] = alpha[m] / beta[m]

    vr = vr.T / np.linalg.norm(vr.T, axis=0)

    return w, vr.transpose(2, 0, 1)
