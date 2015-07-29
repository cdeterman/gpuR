
// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/eigen_helpers.hpp"

// ViennaCL headers
#include "gpuR/vcl_rowMeans.hpp"

using namespace Rcpp;

//[[Rcpp::export]]
void cpp_vienna_fgpuMatrix_rowmean(
    SEXP ptrA_, SEXP ptrC_)
{
    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapVec<float> Cm(ptrC->ptr(), ptrC->length());

    cpp_vienna_rowmean<float>(Am, Cm);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_rowmean(
    SEXP ptrA_, SEXP ptrC_)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapVec<double> Cm(ptrC->ptr(), ptrC->length());

    cpp_vienna_rowmean<double>(Am, Cm);
}
