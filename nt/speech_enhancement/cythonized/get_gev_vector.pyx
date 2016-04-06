# encoding: utf-8
# cython: profile=True
# distutils: extra_compile_args = -fopenmp -O3
# distutils: extra_link_args = -fopenmp
# filename: get_gev_vector.pyx





import numpy as np
cimport numpy as np
from scipy.linalg.cython_lapack cimport zhegvd
cimport cython
from cython.parallel import parallel, prange
cimport openmp

#http://stackoverflow.com/questions/18593308/tips-for-optimising-code-in-cython
# importing math functions from a C-library (faster than numpy)
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI


from time import time
# cdef double first_time = 0
#
# def timeit(f, label):
#     global first_time
#     t = time()
#     f(1.0, 2.0, 10**7)
#     cdef double elapsed = time() - t
#     if first_time == 0:
#         first_time = elapsed
#     print(label, elapsed, (100*elapsed/first_time), '% or', first_time/elapsed, 'x')

cdef double first_time = time()

cdef class Timer:

    cdef double first_time

    def __cinit__(self):
        self.first_time = time()

    name_elapsed_times = list()

    cdef stamp(self, name):

        cdef double elapsed = time() - self.first_time
        print('Time: ', elapsed)
        self.name_elapsed_times.append([name, elapsed])
        self.first_time = time()

    def __str__(self):
        sum = np.sum([e for n, e in self.name_elapsed_times])
        string = ''
        for n, e in self.name_elapsed_times:
            string += '{} % <- {} \n'.format(e/sum*100, n)
        return string



#@cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _c_get_gev_vector(np.ndarray[complex, ndim=3] target_psd_matrix,
                       np.ndarray[complex, ndim=3] noise_psd_matrix):
    """
    Returns the GEV beamforming vector.

    :param target_psd_matrix: Target PSD matrix
        with shape (bins, sensors, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (bins, sensors)
    """

    # Get the dimensions
    cdef size_t sensors = target_psd_matrix.shape[0]
    cdef size_t bins = target_psd_matrix.shape[2]
    cdef int N = target_psd_matrix.shape[0]
    cdef int LDA = target_psd_matrix.shape[1]
    cdef int LDB = noise_psd_matrix.shape[1]


    cdef np.ndarray[complex, ndim=3] a = target_psd_matrix[:]
    cdef np.ndarray[complex, ndim=3] b = noise_psd_matrix[:]
    cdef complex[:, :, :] a_view = a
    cdef complex[:, :, :] b_view = b
    cdef size_t f
    cdef char JOBZ = 'V'
    cdef char UPLO = 'L'
    cdef int ITYPE = 1
    cdef int tmp_size = -1
    cdef int INFO = 0

    # Get the size of the workspace
    cdef complex* a_ptr = &a_view[0, 0, 0]
    cdef complex* b_ptr = &b_view[0, 0, 0]
    cdef np.ndarray[complex, ndim=1] work_tmp = np.empty(2, dtype=np.complex)
    cdef np.ndarray[double, ndim=1] w = np.empty(N, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] rwork_tmp = np.empty(2, dtype=np.float64)
    cdef np.ndarray[int, ndim=1] iwork_tmp = np.empty(2, dtype=np.int32)

    cdef double* w_ptr = <double*> w.data
    cdef complex* work_ptr = <complex*> work_tmp.data
    cdef double* rwork_ptr = <double*> rwork_tmp.data
    cdef int* iwork_ptr = <int*> iwork_tmp.data

    zhegvd(&ITYPE, &JOBZ, &UPLO, &N, a_ptr, &LDA, b_ptr, &LDB, w_ptr, work_ptr,
           &tmp_size, rwork_ptr, &tmp_size, iwork_ptr, &tmp_size, &INFO)

    cdef int work_tmp_size = np.abs(work_tmp[0]).astype(np.int)
    cdef int rwork_tmp_size = np.abs(rwork_tmp[0]).astype(np.int)
    cdef int iwork_tmp_size = np.abs(iwork_tmp[0]).astype(np.int)

    # Create workspace
    cdef np.ndarray[complex, ndim=1] work = np.empty(work_tmp_size,
                                                     dtype=np.complex, order='F')
    cdef np.ndarray[double, ndim=1] rwork = np.empty(rwork_tmp_size,
                                                      dtype=np.float64, order='F')
    cdef np.ndarray[int, ndim=1] iwork = np.empty(iwork_tmp_size,
                                                      dtype=np.int32, order='F')
    work_ptr = <complex*> work.data
    rwork_ptr = <double*> rwork.data
    iwork_ptr = <int*> iwork.data

    cdef np.ndarray[int, ndim=1] max_idx = np.empty(bins, dtype=np.int32)
    cdef int[:] max_idx_view = max_idx
    cdef double[:] w_view = w
    cdef size_t w_idx
    cdef double w_max = 0

    for f in range(bins):
        a_ptr = &a_view[0, 0, f]
        b_ptr = &b_view[0, 0, f]
        zhegvd(&ITYPE, &JOBZ, &UPLO, &N, a_ptr, &LDA,
               b_ptr, &LDB, w_ptr, work_ptr, &work_tmp_size, rwork_ptr,
               &rwork_tmp_size, iwork_ptr, &iwork_tmp_size, &INFO)
        w_max = 0
        if INFO != 0:
            if INFO < 0:
                raise ValueError('Value {} has an illegal value for '
                                 'frequency {}'.format(-INFO, f))
            elif INFO > 0 and INFO < sensors:
                raise ValueError('Algorithm failed to compute an eigenvalue '
                                 'while working on the submatrix lying in rows '
                                 'and columns {i}/({n}+1) '
                                 'through mod({i},{n}+1)'.format(
                    i=INFO, n=sensors))
            else:
                raise ValueError('the leading minor of order {i} of B is not '
                                 'positive definite. The factorization of B '
                                 'could not be completed and no eigenvalues '
                                 'or eigenvectors were computed for '
                                 'frequency {f}'.format(
                    i=INFO-sensors, f=f
                ))
        for w_idx in range(sensors):
            if fabs(w_ptr[w_idx]) > w_max:
                max_idx[f] = w_idx
                w_max = fabs(w_ptr[w_idx])

    return a[:, max_idx, range(bins)].T.conj()

#@cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _c_get_gev_vector_v2(np.ndarray[complex, ndim=3] target_psd_matrix,
                       np.ndarray[complex, ndim=3] noise_psd_matrix):
    """
    Returns the GEV beamforming vector.

    :param target_psd_matrix: Target PSD matrix
        with shape (bins, sensors, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (bins, sensors)
    """

    # Get the dimensions


    t = Timer()

    cdef size_t sensors = target_psd_matrix.shape[0]
    cdef size_t bins = target_psd_matrix.shape[2]
    cdef int N = target_psd_matrix.shape[0]
    cdef int LDA = target_psd_matrix.shape[1]
    cdef int LDB = noise_psd_matrix.shape[1]

    # t.stamp('Init cdefs')

    cdef np.ndarray[complex, ndim=3] a = target_psd_matrix[:]
    cdef np.ndarray[complex, ndim=3] b = noise_psd_matrix[:]

    # t.stamp('Mem view psds')

    cdef complex[:, :, :] a_view = a
    cdef complex[:, :, :] b_view = b

    # t.stamp('Mem view a,b')

    cdef size_t f
    cdef char JOBZ = 'V'
    cdef char UPLO = 'L'
    cdef int ITYPE = 1
    cdef int tmp_size = -1
    cdef int INFO = 0

    # t.stamp('Init vars')


    # Get the size of the workspace
    cdef complex* a_ptr = &a_view[0, 0, 0]
    cdef complex* b_ptr = &b_view[0, 0, 0]

    # t.stamp('Get ptr')

    cdef np.ndarray[complex, ndim=1] work_tmp = np.empty(2, dtype=np.complex)
    cdef np.ndarray[double, ndim=1] w = np.empty(N, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] rwork_tmp = np.empty(2, dtype=np.float64)
    cdef np.ndarray[int, ndim=1] iwork_tmp = np.empty(2, dtype=np.int32)

    # t.stamp('alloc')


    cdef double* w_ptr = <double*> w.data
    cdef complex* work_ptr = <complex*> work_tmp.data
    cdef double* rwork_ptr = <double*> rwork_tmp.data
    cdef int* iwork_ptr = <int*> iwork_tmp.data

    # t.stamp('get prt')

    zhegvd(&ITYPE, &JOBZ, &UPLO, &N, a_ptr, &LDA, b_ptr, &LDB, w_ptr, work_ptr,
           &tmp_size, rwork_ptr, &tmp_size, iwork_ptr, &tmp_size, &INFO)

    # t.stamp('zhegvd')

    cdef int work_tmp_size = <int> abs(work_tmp[0])
    cdef int rwork_tmp_size = <int> abs(rwork_tmp[0])
    cdef int iwork_tmp_size = <int> abs(iwork_tmp[0])

    # t.stamp('abs')

    # Create workspace
    cdef np.ndarray[complex, ndim=1] work = np.empty(work_tmp_size,
                                                     dtype=np.complex, order='F')
    cdef np.ndarray[double, ndim=1] rwork = np.empty(rwork_tmp_size,
                                                      dtype=np.float64, order='F')
    cdef np.ndarray[int, ndim=1] iwork = np.empty(iwork_tmp_size,
                                                      dtype=np.int32, order='F')

    # t.stamp('alloc')

    work_ptr = <complex*> work.data
    rwork_ptr = <double*> rwork.data
    iwork_ptr = <int*> iwork.data

    # t.stamp('get ptr')

    cdef np.ndarray[int, ndim=1] max_idx = np.empty(bins, dtype=np.int32)

    # t.stamp('alloc')

    cdef int[:] max_idx_view = max_idx
    cdef int* max_idx_ptr = <int*> max_idx.data
    cdef double[:] w_view = w
    cdef size_t w_idx
    cdef double w_max = 0

    # t.stamp('etc')

    cdef complex* a_ptr_global = &a_view[0, 0, 0]
    cdef complex* b_ptr_global = &b_view[0, 0, 0]

    # t.stamp('view')

    cdef int sensors_square = sensors * sensors

    t.stamp('head')

    for f in range(bins):
        # a_ptr = &a_view[0, 0, f]
        # b_ptr = &b_view[0, 0, f]
        a_ptr = a_ptr_global + sensors_square * f
        b_ptr = b_ptr_global + sensors_square * f
        zhegvd(&ITYPE, &JOBZ, &UPLO, &N, a_ptr, &LDA,
               b_ptr, &LDB, w_ptr, work_ptr, &work_tmp_size, rwork_ptr,
               &rwork_tmp_size, iwork_ptr, &iwork_tmp_size, &INFO)
        if INFO != 0:
            if INFO < 0:
                raise ValueError('Value {} has an illegal value for '
                                 'frequency {}'.format(-INFO, f))
            elif INFO > 0 and INFO < sensors:
                raise ValueError('Algorithm failed to compute an eigenvalue '
                                 'while working on the submatrix lying in rows '
                                 'and columns {i}/({n}+1) '
                                 'through mod({i},{n}+1)'.format(
                    i=INFO, n=sensors))
            else:
                raise ValueError('the leading minor of order {i} of B is not '
                                 'positive definite. The factorization of B '
                                 'could not be completed and no eigenvalues '
                                 'or eigenvectors were computed for '
                                 'frequency {f}'.format(
                    i=INFO-sensors, f=f
                ))
        w_max = 0
        #for w_idx in range(sensors):
        #    if fabs(w_ptr[w_idx]) > w_max:
        #        max_idx_ptr[f] = w_idx
        #        w_max = fabs(w_ptr[w_idx])


    t.stamp('loop')

    print(t)

    return a[:, -1, range(bins)].T.conj()
    # return a[:, max_idx, range(bins)].T.conj()

@cython.boundscheck(False)
@cython.wraparound(False)
def _c_get_gev_vector_parallel(np.ndarray[complex, ndim=3] target_psd_matrix,
                       np.ndarray[complex, ndim=3] noise_psd_matrix):
    """
    Returns the GEV beamforming vector.

    :param target_psd_matrix: Target PSD matrix
        with shape (bins, sensors, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (bins, sensors)
    """

    #t = Timer()


    # Get the dimensions
    cdef size_t sensors = target_psd_matrix.shape[0]
    cdef size_t bins = target_psd_matrix.shape[2]
    cdef int N = target_psd_matrix.shape[0]
    cdef int LDA = target_psd_matrix.shape[1]
    cdef int LDB = noise_psd_matrix.shape[1]


    cdef np.ndarray[complex, ndim=3] a = target_psd_matrix[:]
    cdef np.ndarray[complex, ndim=3] b = noise_psd_matrix[:]
    cdef complex[:, :, :] a_view = a
    cdef complex[:, :, :] b_view = b
    cdef size_t f
    cdef char JOBZ = 'V'
    cdef char UPLO = 'L'
    cdef int ITYPE = 1
    cdef int tmp_size = -1
    cdef int INFO = 0

    # Get the size of the workspace
    cdef complex* a_ptr = &a_view[0, 0, 0]
    cdef complex* b_ptr = &b_view[0, 0, 0]
    cdef np.ndarray[complex, ndim=1] work_tmp = np.empty(2, dtype=np.complex)
    cdef np.ndarray[double, ndim=1] w = np.empty(N, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] rwork_tmp = np.empty(2, dtype=np.float64)
    cdef np.ndarray[int, ndim=1] iwork_tmp = np.empty(2, dtype=np.int32)

    cdef double* w_ptr = <double*> w.data
    cdef complex* work_ptr = <complex*> work_tmp.data
    cdef double* rwork_ptr = <double*> rwork_tmp.data
    cdef int* iwork_ptr = <int*> iwork_tmp.data

    zhegvd(&ITYPE, &JOBZ, &UPLO, &N, a_ptr, &LDA, b_ptr, &LDB, w_ptr, work_ptr,
           &tmp_size, rwork_ptr, &tmp_size, iwork_ptr, &tmp_size, &INFO)

    cdef int work_tmp_size = np.abs(work_tmp[0]).astype(np.int)
    cdef int rwork_tmp_size = np.abs(rwork_tmp[0]).astype(np.int)
    cdef int iwork_tmp_size = np.abs(iwork_tmp[0]).astype(np.int)

    # Create workspace
    cdef np.ndarray[complex, ndim=2] work = np.empty((work_tmp_size, bins),
                                                     dtype=np.complex, order='F')
    cdef np.ndarray[double, ndim=2] rwork = np.empty((rwork_tmp_size, bins),
                                                      dtype=np.float64, order='F')
    cdef np.ndarray[int, ndim=2] iwork = np.empty((iwork_tmp_size, bins),
                                                      dtype=np.int32, order='F')
    cdef np.ndarray[double, ndim=2] w_work = np.empty((N, bins),
                                                      dtype=np.float64, order='F')
    work_ptr = <complex*> work.data
    rwork_ptr = <double*> rwork.data
    iwork_ptr = <int*> iwork.data
    w_ptr = <double*> w_work.data

    cdef complex[:, :] work_view = work
    cdef double[:, :] rwork_view = rwork
    cdef int[:, :] iwork_view = iwork
    cdef double[:, :] w_work_view = w_work

    cdef np.ndarray[int, ndim=1] max_idx = np.empty(bins, dtype=np.int32)
    cdef int[:] max_idx_view = max_idx
    cdef int* max_idx_ptr = <int*> max_idx.data
    cdef double[:] w_view = w
    cdef size_t w_idx

    # cdef double arr[N*bins]

    cdef double w_max = 0
    cdef Py_ssize_t offset

    #t.stamp('head')

    #openmp.omp_set_dynamic(5)
    #openmp.omp_sched_dynamic(1)
    #print('Threads: ', openmp.omp_get_num_threads())
    for f in prange(bins, nogil=True):
        INFO = 0
        a_ptr = &a_view[0, 0, f]
        b_ptr = &b_view[0, 0, f]

        # offset = work_tmp_size * f

        zhegvd(&ITYPE, &JOBZ, &UPLO, &N,
               &a_view[0, 0, f], &LDA,
               &b_view[0, 0, f], &LDB,
               w_ptr + N * f,
               work_ptr + work_tmp_size * f, &work_tmp_size,
               rwork_ptr + rwork_tmp_size * f, &rwork_tmp_size,
               iwork_ptr + iwork_tmp_size * f, &iwork_tmp_size,
               &INFO)

        if INFO != 0:
            if INFO < 0:
                with gil:
                    raise ValueError('Value {} has an illegal value for '
                                 'frequency {}'.format(-INFO, f))
            elif INFO > 0 and INFO < sensors:
                with gil:
                    raise ValueError('Algorithm failed to compute an eigenvalue '
                                     'while working on the submatrix lying in rows '
                                     'and columns {i}/({n}+1) '
                                     'through mod({i},{n}+1)'.format(
                        i=INFO, n=sensors))
            else:
                with gil:
                    raise ValueError('the leading minor of order {i} of B is not '
                                     'positive definite. The factorization of B '
                                     'could not be completed and no eigenvalues '
                                     'or eigenvectors were computed for '
                                     'frequency {f}'.format(
                        i=INFO-sensors, f=f
                ))

    #print('zets')

    #for f in prange(bins, nogil=True):
    #    w_max = 0 # with this line the variable will be local
    #    #https://groups.google.com/forum/#!topic/cython-users/dCqd70kz_1U
    #    for w_idx in range(sensors):
    #        if fabs(w_ptr[N * f + w_idx]) > w_max:
    #            max_idx_ptr[f] = w_idx
    #            w_max = fabs(w_ptr[N * f + w_idx])

    #print('zetsas')
    # string = ''
    # for i in range(bins):
    #     string += str(max_idx_ptr[i])
    #
    # print(string)
    # print('zetsasas')

    return target_psd_matrix[:, -1, :].T.conj()

#    return a[:, max_idx, range(bins)].T.conj()