#ifndef EIGEN_HELPERS
#define EIGEN_HELPERS

#include <RcppEigen.h>

#include "dynEigen.hpp"


template<class T>
using MapMat = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >;

//template<typename T>
//struct MapMat
//{
//    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Type;
//};

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
MapMat<T> XPtrToSEXP(SEXP ptrA_)
{
    Rcpp::XPtr<dynEigen<T> > ptrA(ptrA_);
    MapMat<T> A = MapMat<T>(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
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
