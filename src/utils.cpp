
#include "gpuR/windows_check.hpp"

#include <RcppEigen.h>

#include "gpuR/dynEigenVec.hpp"

using namespace Rcpp;

template <typename T>
int 
cpp_ncol(SEXP ptrA_)
{       
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > pMat(ptrA_);
    return pMat->cols();
}

template <typename T>
int 
cpp_nrow(SEXP ptrA_)
{
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > pMat(ptrA_);
    return pMat->rows();
}

template <typename T>
int 
cpp_gpuVec_size(SEXP ptrA_)
{
    XPtr<dynEigenVec<T> > pMat(ptrA_);
    return pMat->length();
}

//template <typename T>
//int cpp_gpuVecSlice_length(SEXP ptrA_)
//{
//    Rcpp::XPtr<dynEigenVec<T> > ptrA(ptrA_);
//    const int len = ptrA->end()+1 - ptrA->start();
//    return len;
//}
//
///*** Slice Vector ***/
//// [[Rcpp::export]]
//int
//cpp_gpuVecSlice_length(SEXP ptrA_, const int type_flag)
//{    
//    switch(type_flag) {
//        case 4:
//            return cpp_gpuVecSlice_length<int>(ptrA_);
//        case 6:
//            return cpp_gpuVecSlice_length<float>(ptrA_);
//        case 8:
//            return cpp_gpuVecSlice_length<double>(ptrA_);
//        default:
//            throw Rcpp::exception("unknown type detected for gpuVectorSlice object!");
//    }
//}

// [[Rcpp::export]]
int 
cpp_dncol(SEXP ptrA)
{
    return cpp_ncol<double>(ptrA);
}

// [[Rcpp::export]]
int 
cpp_fncol(SEXP ptrA)
{
    return cpp_ncol<float>(ptrA);
}

// [[Rcpp::export]]
int 
cpp_incol(SEXP ptrA)
{
    return cpp_ncol<int>(ptrA);
}

// [[Rcpp::export]]
int 
cpp_dnrow(SEXP ptrA)
{
    return cpp_nrow<double>(ptrA);
}

// [[Rcpp::export]]
int 
cpp_fnrow(SEXP ptrA)
{
    return cpp_nrow<float>(ptrA);
}

// [[Rcpp::export]]
int 
cpp_inrow(SEXP ptrA)
{
    return cpp_nrow<int>(ptrA);
}

/*** gpuVector size ***/

// [[Rcpp::export]]
int 
cpp_dgpuVec_size(SEXP ptrA)
{
    return cpp_gpuVec_size<double>(ptrA);
}

// [[Rcpp::export]]
int 
cpp_fgpuVec_size(SEXP ptrA)
{
    return cpp_gpuVec_size<float>(ptrA);
}

// [[Rcpp::export]]
int 
cpp_igpuVec_size(SEXP ptrA)
{
    return cpp_gpuVec_size<int>(ptrA);
}
