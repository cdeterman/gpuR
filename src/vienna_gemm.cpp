
// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "eigen_helpers.hpp"

// ViennaCL headers
#include "vcl_gemm.hpp"

using namespace Rcpp;

//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_dgemm(SEXP ptrA_, 
                                SEXP ptrB_,
                                SEXP ptrC_)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am = MapMat<double>(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Bm = MapMat<double>(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<double> Cm = MapMat<double>(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_arma_vienna_gemm(Am, Bm, Cm);
}

//[[Rcpp::export]]
SEXP cpp_vienna_gpuMatrix_sgemm(SEXP ptrA_, 
                                SEXP ptrB_, 
                                SEXP ptrC_)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am = MapMat<float>(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Bm = MapMat<float>(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<float> Cm = MapMat<float>(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_arma_vienna_gemm(Am, Bm, Cm);
}
