#ifndef EIGEN_HELPERS
#define EIGEN_HELPERS

#include <RcppEigen.h>

#include "dynEigen.hpp"


// Would very much prefer to use this new C++11 syntax
// but the Travis-CI g++ always defaults to use C++0x which is 
// failing with this alias typedef so using the hideous struct below
// requiring the terrible MapMat<T>::Type syntax which also requires
// a typename declariation in each instance

//template<class T>
//using MapMat = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >;

template<class T>
struct MapMat
{
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Type;
};

template <typename T>
SEXP sexpToXptr(SEXP A)
{
    dynEigen<T> *C = new dynEigen<T>(A);
    Rcpp::XPtr<dynEigen<T> > pMat(C);
    return pMat;
}

// convert an XPtr back to a MapMat object to ultimately 
// be returned as a SEXP object
template <typename T>
typename MapMat<T>::Type XPtrToSEXP(SEXP ptrA_)
{
    Rcpp::XPtr<dynEigen<T> > ptrA(ptrA_);
    typename MapMat<T>::Type A(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    return A;
}

// create an empty eigen matrix
template <typename T>
SEXP emptyXptr(int nr, int nc)
{
    dynEigen<T> *C = new dynEigen<T>(nr, nc);
    Rcpp::XPtr<dynEigen<T> > pMat(C);
    return pMat;
}
 
#endif
