#pragma once
#ifndef EIGEN_HELPERS
#define EIGEN_HELPERS

#include <RcppEigen.h>

//#include "eigen_templates.hpp"

using namespace Rcpp;

// convert SEXP Matrix to Eigen matrix
template <typename T>
SEXP sexpToEigenXptr(SEXP A, const int nr, const int nc)
{    
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> *eigen_mat = new Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(nr, nc);
    *eigen_mat = as<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > pMat(eigen_mat);
    return pMat;
}

// convert SEXP Vector to Eigen Vector (i.e. 1 column matrix)
template <typename T>
SEXP 
sexpVecToEigenVecXptr(SEXP A, const int size)
{
    Eigen::Matrix<T, Eigen::Dynamic, 1> *eigen_mat = new Eigen::Matrix<T, Eigen::Dynamic, 1>(size, 1);
    *eigen_mat = as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > pMat(eigen_mat);
    return pMat;
}

// convert an XPtr back to a MapMat object to ultimately 
// be returned as a SEXP object
template <typename T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > 
EigenXPtrToMapEigen(SEXP ptrA_)
{    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > pMat(ptrA_);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > MapMat(pMat->data(), pMat->rows(), pMat->cols());
    return MapMat;
}

// convert an XPtr back to a MapVec object to ultimately 
// be returned as a SEXP object
template <typename T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > 
EigenVecXPtrToMapEigenVec(SEXP ptrA_)
{
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > pMat(ptrA_);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > MapMat(pMat->data(), pMat->rows(), 1);
    return MapMat;
}

// create an empty eigen matrix
template <typename T>
SEXP emptyEigenXptr(int nr, int nc)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> *eigen_mat = new Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(nr, nc);
    *eigen_mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(nr, nc);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > pMat(eigen_mat);
    return pMat;
}

// create an empty eigen vector
template <typename T>
SEXP 
emptyEigenVecXptr(const int size)
{
    Eigen::Matrix<T, Eigen::Dynamic, 1> *eigen_mat = new Eigen::Matrix<T, Eigen::Dynamic, 1>(size, 1);
    *eigen_mat = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(size, 1);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > pMat(eigen_mat);
    return pMat;
}


template <typename T>
void
SetMatRow(SEXP data, const int idx, SEXP value)
{    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > A = EigenXPtrToMapEigen<T>(data);
    A.row(idx-1) = as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(value);
}


template <typename T>
void
SetMatCol(SEXP data, const int idx, SEXP value)
{    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > A = EigenXPtrToMapEigen<T>(data);
    A.col(idx-1) = as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(value);
}

template <typename T>
void
SetMatElement(SEXP data, const int nr, const int nc, SEXP value)
{    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > A = EigenXPtrToMapEigen<T>(data);
    A(nr-1, nc-1) = as<T>(value);
}

template <typename T>
SEXP
GetMatRow(const SEXP data, const int idx)
{    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > A = EigenXPtrToMapEigen<T>(data);
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am = A.row(idx-1);
    return(wrap(Am));
}

template <typename T>
SEXP
GetMatCol(const SEXP data, const int idx)
{    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > A = EigenXPtrToMapEigen<T>(data);
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am = A.col(idx-1);
    return(wrap(Am));
}

template <typename T>
SEXP
GetMatElement(const SEXP data, const int nr, const int nc)
{    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > A = EigenXPtrToMapEigen<T>(data);
    T value = A(nr-1, nc-1);
    return(wrap(value));
}
 
#endif
