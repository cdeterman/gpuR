#include <RcppEigen.h>

#include "gpuR/eigen_helpers.hpp"

using namespace Rcpp;

template <typename T>
int cpp_ncol(SEXP ptrA_)
{
    Rcpp::XPtr<dynEigen<T> > ptrA(ptrA_);
    return ptrA->ncol();
}

template <typename T>
int cpp_nrow(SEXP ptrA_)
{
    Rcpp::XPtr<dynEigen<T> > ptrA(ptrA_);
    return ptrA->nrow();
}

template <typename T>
int cpp_gpuVec_size(SEXP ptrA_)
{
    Rcpp::XPtr<dynEigenVec<T> > ptrA(ptrA_);
    return ptrA->length();
}

// [[Rcpp::export]]
int cpp_dncol(SEXP ptrA)
{
    return cpp_ncol<double>(ptrA);
}

// [[Rcpp::export]]
int cpp_fncol(SEXP ptrA)
{
    return cpp_ncol<float>(ptrA);
}

// [[Rcpp::export]]
int cpp_incol(SEXP ptrA)
{
    return cpp_ncol<int>(ptrA);
}

// [[Rcpp::export]]
int cpp_dnrow(SEXP ptrA)
{
    return cpp_nrow<double>(ptrA);
}

// [[Rcpp::export]]
int cpp_fnrow(SEXP ptrA)
{
    return cpp_nrow<float>(ptrA);
}

// [[Rcpp::export]]
int cpp_inrow(SEXP ptrA)
{
    return cpp_nrow<int>(ptrA);
}

/*** gpuVector size ***/

// [[Rcpp::export]]
int cpp_dgpuVec_size(SEXP ptrA)
{
    return cpp_gpuVec_size<double>(ptrA);
}

// [[Rcpp::export]]
int cpp_fgpuVec_size(SEXP ptrA)
{
    return cpp_gpuVec_size<float>(ptrA);
}

// [[Rcpp::export]]
int cpp_igpuVec_size(SEXP ptrA)
{
    return cpp_gpuVec_size<int>(ptrA);
}
