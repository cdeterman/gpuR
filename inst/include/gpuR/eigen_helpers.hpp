#ifndef EIGEN_HELPERS
#define EIGEN_HELPERS

#include <RcppEigen.h>

#include "dynEigen.hpp"
#include "dynEigenVec.hpp"


// Would very much prefer to use this new C++11 syntax
// but the Travis-CI g++ always defaults to use C++0x which is 
// failing with this alias typedef so using the hideous struct below
// requiring the terrible MapMat<T>::Type syntax which also requires
// a typename declaration in each instance

template<class T>
using MapMat = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >;


template<class T>
using MapVec = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> >;

//template<class T>
//struct MapMat
//{
//    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Type;
//};
//
//template<class T>
//struct MapVec
//{
//    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Type;
//};

// convert SEXP Matrix to Eigen matrix
template <typename T>
SEXP sexpToXptr(SEXP A)
{
    dynEigen<T> *C = new dynEigen<T>(A);
    Rcpp::XPtr<dynEigen<T> > pMat(C);
    return pMat;
}

// convert SEXP Vector to Eigen Vector (i.e. 1 column matrix)
template <typename T>
SEXP sexpVecToXptr(SEXP A)
{
    dynEigenVec<T> *C = new dynEigenVec<T>(A);
    Rcpp::XPtr<dynEigenVec<T> > pVec(C);
    return pVec;
}

// convert SEXP Vector to Eigen matrix
template <typename T>
SEXP sexpVecToMatXptr(SEXP A, int nr, int nc)
{
    dynEigen<T> *C = new dynEigen<T>(A, nr, nc);
    Rcpp::XPtr<dynEigen<T> > pMat(C);
    return pMat;
}

// convert an XPtr back to a MapMat object to ultimately 
// be returned as a SEXP object
template <typename T>
MapMat<T> XPtrToSEXP(SEXP ptrA_)
{
    Rcpp::XPtr<dynEigen<T> > ptrA(ptrA_);
    MapMat<T> A(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    return A;
}


// convert an XPtr back to a MapVec object to ultimately 
// be returned as a SEXP object
template <typename T>
MapVec<T> XPtrToVecSEXP(SEXP ptrA_)
{
    Rcpp::XPtr<dynEigenVec<T> > ptrA(ptrA_);
    MapVec<T> A(ptrA->ptr(), ptrA->length());
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

// create an empty eigen vector
template <typename T>
SEXP emptyVecXptr(int size)
{
    dynEigenVec<T> *C = new dynEigenVec<T>(size);
    Rcpp::XPtr<dynEigenVec<T> > pVec(C);
    return pVec;
}
 
#endif
