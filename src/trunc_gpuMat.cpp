/* Truncating a gpuMatrix object
 * The point here is to simply return the top-left corner of
 * a given matrix.  This way we avoid overflow in the console
 * and the user is still able to view their data.  The option
 * to return the full matrix is possible by converting back
 * to an R matrix object with the '[' method.
 */

#include <RcppEigen.h>

#include "eigen_helpers.hpp"

using namespace Rcpp;

template<typename T>
SEXP trunc_mat(SEXP ptrA_, int nr, int nc)
{
    Rcpp::XPtr<dynEigen<T> > ptrA(ptrA_);
    typename MapMat<T>::Type Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    return wrap(Am.topLeftCorner(nr, nc));
}

// [[Rcpp::export]]
SEXP truncIntgpuMat(SEXP ptrA_, int nr, int nc)
{
    return trunc_mat<int>(ptrA_, nr, nc);   
}

// [[Rcpp::export]]
SEXP truncFloatgpuMat(SEXP ptrA_, int nr, int nc)
{
    return trunc_mat<float>(ptrA_, nr, nc);   
}

// [[Rcpp::export]]
SEXP truncDoublegpuMat(SEXP ptrA_, int nr, int nc)
{
    return trunc_mat<double>(ptrA_, nr, nc);   
}
