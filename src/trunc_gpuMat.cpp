/* Truncating a gpuMatrix object
 * The point here is to simply return the top-left corner of
 * a given matrix.  This way we avoid overflow in the console
 * and the user is still able to view their data.  The option
 * to return the full matrix is possible by converting back
 * to an R matrix object with the '[' method.
 */
 
#include "gpuR/windows_check.hpp"
#include "gpuR/dynEigenMat.hpp"

#include <RcppEigen.h>

using namespace Rcpp;

template<typename T>
SEXP 
trunc_mat(SEXP ptrA_, int nr, int nc)
{    
//    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > pMat(ptrA_);
//    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(pMat->data(), pMat->rows(), pMat->cols());
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > refA = ptrA->data();
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::OuterStride<> > Am(
        refA.data(), refA.rows(), refA.cols(),
        Eigen::OuterStride<>(refA.outerStride())
    );
    
    return wrap(Am.topLeftCorner(nr, nc));
}

// [[Rcpp::export]]
SEXP 
truncIntgpuMat(SEXP ptrA_, int nr, int nc)
{
    return trunc_mat<int>(ptrA_, nr, nc);   
}

// [[Rcpp::export]]
SEXP 
truncFloatgpuMat(SEXP ptrA_, int nr, int nc)
{
    return trunc_mat<float>(ptrA_, nr, nc);   
}

// [[Rcpp::export]]
SEXP 
truncDoublegpuMat(SEXP ptrA_, int nr, int nc)
{
    return trunc_mat<double>(ptrA_, nr, nc);   
}
