
// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/eigen_helpers.hpp"

// ViennaCL headers
#include "gpuR/vcl_crossprod.hpp"

using namespace Rcpp;

//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_dcrossprod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_crossprod<double>(Am, Bm, Cm);
}

//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_scrossprod(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_crossprod<float>(Am, Bm, Cm);
}

//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_dtcrossprod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_tcrossprod<double>(Am, Bm, Cm);
}

//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_stcrossprod(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_tcrossprod<float>(Am, Bm, Cm);
}

