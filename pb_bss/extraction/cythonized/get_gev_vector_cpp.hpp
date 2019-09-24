#include "tbb/tbb.h"
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <complex> // for std::complex<double>
#include <assert.h>

extern "C"{
void zhegvd_(int *ITYPE,
            char *JOBZ,
            char *UPLO,
            int *N,
            std::complex<double> *A,
            int *LDA,
            std::complex<double> *B,
            int *LDB,
            double *W,
            std::complex<double> *WORK,
            int *LWORK,
            double *RWORK,
            int *LRWORK,
            int *IWORK,
            int *LIWORK,
            int *INFO
           );
}
void call_zhegvd(std::complex<double> *a, std::complex<double> *b, const int mat_size, const int F){
    // int mat_size = 6;

    int ITYPE = 1;
    char JOBZ = 'V';
    char UPLO = 'L';
    int N = mat_size;
    int LDA = mat_size;
    int LDB = mat_size;
    int tmp_size = -1;
    int INFO = 0;

    //std::vector<std::complex<double>> a(mat_size*mat_size);
    //std::vector<std::complex<double>> b(mat_size*mat_size);

    std::complex<double> work_tmp[2];
    double w[N];
    double rwork_tmp[2];
    int iwork_tmp[2];

    zhegvd_(&ITYPE, &JOBZ, &UPLO,
            &N, a, &LDA, b, &LDB, w,
            work_tmp, &tmp_size,
            rwork_tmp, &tmp_size,
            iwork_tmp, &tmp_size,
            &INFO);

    int work_size = int(fabs(work_tmp[0]));
    int rwork_size = int(fabs(rwork_tmp[0]));
    int iwork_size = int(fabs(iwork_tmp[0]));

    std::complex<double> work[work_size];
    double rwork[rwork_size];
    int iwork[iwork_size];

    int step = mat_size*mat_size;
    for(int f = 0; f < F; f++){
        int local_step = step * f;
        zhegvd_(&ITYPE, &JOBZ, &UPLO, &N,
               a+local_step, &LDA, b+local_step, &LDB, w,
               work, &work_size,
               rwork, &rwork_size,
               iwork, &iwork_size,
               &INFO);
        if (INFO != 0){
            if(INFO < 0){
                std::cout << "Value " << -INFO << " has an illegal value for frequency " << f << std::endl;
                assert(false && __file__ + __line__);
            }else if(INFO > 0 and INFO < mat_size){
                std::cout << "Algorithm failed to compute an eigenvalue "
                << "while working on the submatrix lying in rows "
                << "and columns " << INFO << " / (" << mat_size << "+1)"
                << "through mod(" << INFO << " / (" << mat_size << "+1)" << std::endl;
                assert(false && __file__ + __line__);
            }else{
                std::cout
                << "the leading minor of order " << INFO << " of B is not "
                << "positive definite. The factorization of B "
                << "could not be completed and no eigenvalues "
                << "or eigenvectors were computed for "
                << "frequency " << f << std::endl;
                assert(false && __file__ + __line__);
            }
        }
        // assert(INFO == 0);
    }

    // return a;
}
