#pragma once
#ifndef EIGEN_HELPERS
#define EIGEN_HELPERS

#include <RcppEigen.h>

#include "gpuR/dynEigenMat.hpp"
#include "gpuR/dynEigenVec.hpp"

using namespace Rcpp;

// convert SEXP Matrix to Eigen matrix
template <typename T>
SEXP 
getRmatEigenAddress(SEXP A, const int nr, const int nc)
{    
//    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> *eigen_mat = new Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(nr, nc);
//    *eigen_mat = as<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A);
//    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > pMat(eigen_mat);
//    return pMat;
    
    dynEigenMat<T> *mat = new dynEigenMat<T>(A);
    Rcpp::XPtr<dynEigenMat<T> > pMat(mat);
    return pMat;
}

// convert SEXP Vector to Eigen Vector (i.e. 1 column matrix)
template <typename T>
SEXP 
sexpVecToEigenVecXptr(SEXP A, const int size)
{
    dynEigenVec<T> *vec = new dynEigenVec<T>(A);
    Rcpp::XPtr<dynEigenVec<T> > pVec(vec);
    return pVec;
}

// convert an XPtr back to a MapMat object to ultimately 
// be returned as a SEXP object
template <typename T>
Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > 
EigenXPtrToMapEigen(SEXP ptrA_)
{    
//    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > pMat(ptrA_);
//    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > MapMat(pMat->data(), pMat->rows(), pMat->cols());
//    return MapMat;
    
    Rcpp::XPtr<dynEigenMat<T> > pMat(ptrA_);
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > MapMat = pMat->data();
    return MapMat;
}

// convert an XPtr back to a MapVec object to ultimately 
// be returned as a SEXP object
template <typename T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > 
EigenVecXPtrToMapEigenVec(SEXP ptrA_)
{
    Rcpp::XPtr<dynEigenVec<T> > pVec(ptrA_);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > MapVec = pVec->data();
    return MapVec;
}

// create an empty eigen matrix
template <typename T>
SEXP emptyEigenXptr(int nr, int nc)
{
//    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> *eigen_mat = new Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(nr, nc);
//    *eigen_mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(nr, nc);
//    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > pMat(eigen_mat);
//    return pMat;
    
//    std::cout << "eigen helpers output" << std::endl;
//    std::cout << nr << nc << std::endl;
    dynEigenMat<T> *mat = new dynEigenMat<T>(nr, nc);
//    std::cout << mat->data() << std::endl;
    Rcpp::XPtr<dynEigenMat<T> > pMat(mat);
    return pMat;
}

// create an empty eigen vector
template <typename T>
SEXP 
emptyEigenVecXptr(const int size)
{    
    dynEigenVec<T> *vec = new dynEigenVec<T>(size);
    Rcpp::XPtr<dynEigenVec<T> > pVec(vec);
    return pVec;
}


template <typename T>
void
SetMatRow(SEXP data, const int idx, SEXP value)
{    
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > A = EigenXPtrToMapEigen<T>(data);
    A.row(idx-1) = as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(value);
}


template <typename T>
void
SetMatCol(SEXP data, const int idx, SEXP value)
{    
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > A = EigenXPtrToMapEigen<T>(data);
    A.col(idx-1) = as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(value);
}

template <typename T>
void
SetMatElement(SEXP data, const int nr, const int nc, SEXP value)
{    
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > A = EigenXPtrToMapEigen<T>(data);
    A(nr-1, nc-1) = as<T>(value);
}

template <typename T>
SEXP
GetMatRow(const SEXP data, const int idx)
{    
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > A = EigenXPtrToMapEigen<T>(data);
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am = A.row(idx-1);
    return(wrap(Am));
}

template <typename T>
SEXP
GetMatCol(const SEXP data, const int idx)
{    
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > A = EigenXPtrToMapEigen<T>(data);
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am = A.col(idx-1);
    return(wrap(Am));
}

template <typename T>
SEXP
GetMatElement(const SEXP data, const int nr, const int nc)
{    
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > A = EigenXPtrToMapEigen<T>(data);
    T value = A(nr-1, nc-1);
    return(wrap(value));
}
 
#endif
