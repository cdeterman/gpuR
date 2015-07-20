
// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/eigen_helpers.hpp"

// ViennaCL headers
#include "gpuR/vcl_axpy.hpp"

using namespace Rcpp;

//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_daxpy(SEXP alpha_, 
                                SEXP ptrA_, 
                                SEXP ptrB_)
{
    const double alpha = as<double>(alpha_);
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrB(ptrB_);
    
    MapMat<double> Am = MapMat<double>(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Bm = MapMat<double>(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    
    cpp_arma_vienna_axpy(alpha, Am, Bm);
}

//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_saxpy(SEXP alpha_, 
                                SEXP ptrA_, 
                                SEXP ptrB_)
{
    const float alpha = as<float>(alpha_);

    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrB(ptrB_);
    
    MapMat<float> Am = MapMat<float>(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Bm = MapMat<float>(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    
    cpp_arma_vienna_axpy(alpha, Am, Bm);
}
