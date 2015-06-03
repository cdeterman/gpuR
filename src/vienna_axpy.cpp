
// Armadillo headers (disable BLAS and LAPACK to avoid linking issues)
#define ARMA_DONT_USE_BLAS
#define ARMA_DONT_USE_LAPACK

// armadillo headers for handling the R input data
#include <RcppArmadillo.h>

// ViennaCL headers
#include "vcl_axpy.hpp"

using namespace Rcpp;

//[[Rcpp::export]]
SEXP cpp_vienna_gpuMatrix_daxpy(SEXP alpha_, SEXP A_, SEXP B_)
{
    const double alpha = as<double>(alpha_);
    const arma::Mat<double> Am = as<arma::Mat<double> >(A_);
    const arma::Mat<double> Bm = as<arma::Mat<double> >(B_);
    
    arma::Mat<double> Cm = cpp_arma_vienna_axpy(alpha, Am, Bm);
    
    return wrap(Cm);
}

//[[Rcpp::export]]
SEXP cpp_vienna_gpuMatrix_saxpy(SEXP alpha_, SEXP A_, SEXP B_)
{
    const float alpha = as<float>(alpha_);
    const arma::Mat<float> Am = as<arma::Mat<float> >(A_);
    const arma::Mat<float> Bm = as<arma::Mat<float> >(B_);
    
    arma::Mat<float> Cm = cpp_arma_vienna_axpy(alpha, Am, Bm);
    
    return wrap(Cm);
}
