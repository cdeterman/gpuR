
// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/eigen_helpers.hpp"

// ViennaCL headers
#include "gpuR/vcl_pmcc.hpp"

using namespace Rcpp;


//[[Rcpp::export]]
void cpp_vienna_fgpuMatrix_pmcc(
    SEXP ptrA_, SEXP ptrB_)
{
    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrB(ptrB_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());

    cpp_vienna_pmcc<float>(Am, Bm);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_pmcc(
    SEXP ptrA_, SEXP ptrB_)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrB(ptrB_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());

    cpp_vienna_pmcc<double>(Am, Bm);
}
