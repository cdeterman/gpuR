
// Armadillo headers (disable BLAS and LAPACK to avoid linking issues)
#define ARMA_DONT_USE_BLAS
#define ARMA_DONT_USE_LAPACK

// armadillo headers for handling the R input data
#include <RcppArmadillo.h>

// ViennaCL headers
#include "vcl_gemm.hpp"

using namespace Rcpp;

//[[Rcpp::export]]
SEXP cpp_vienna_gpuMatrix_dgemm(SEXP A_, SEXP B_)
{
    const arma::Mat<double> Am = as<arma::Mat<double> >(A_);
    const arma::Mat<double> Bm = as<arma::Mat<double> >(B_);
    
    arma::Mat<double> Cm = cpp_arma_vienna_gemm(Am, Bm);
    
    return wrap(Cm);
}

//[[Rcpp::export]]
SEXP cpp_vienna_gpuMatrix_sgemm(SEXP A_, SEXP B_)
{
    const arma::Mat<float> Am = as<arma::Mat<float> >(A_);
    const arma::Mat<float> Bm = as<arma::Mat<float> >(B_);
    
    arma::Mat<float> Cm = cpp_arma_vienna_gemm(Am, Bm);
    
    return wrap(Cm);
}
