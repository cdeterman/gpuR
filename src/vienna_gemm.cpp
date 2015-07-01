
// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/eigen_helpers.hpp"

// ViennaCL headers
#include "gpuR/vcl_gemm.hpp"

using namespace Rcpp;

//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_dgemm(SEXP ptrA_, 
                                SEXP ptrB_,
                                SEXP ptrC_)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    typename MapMat<double>::Type Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    typename MapMat<double>::Type Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    typename MapMat<double>::Type Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_arma_vienna_gemm<double>(Am, Bm, Cm);
}

//[[Rcpp::export]]
SEXP cpp_vienna_gpuMatrix_sgemm(SEXP ptrA_, 
                                SEXP ptrB_, 
                                SEXP ptrC_)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    typename MapMat<float>::Type Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    typename MapMat<float>::Type Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    typename MapMat<float>::Type Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_arma_vienna_gemm<float>(Am, Bm, Cm);
}
