
// Armadillo headers (disable BLAS and LAPACK to avoid linking issues)
#define ARMA_DONT_USE_BLAS
#define ARMA_DONT_USE_LAPACK

// armadillo headers for handling the R input data
#include <RcppArmadillo.h>

// ViennaCL headers
#include "vcl_vec_axpy.hpp"

using namespace Rcpp;

//[[Rcpp::export]]
SEXP cpp_vienna_gpuVector_daxpy(SEXP alpha_, SEXP A_, SEXP B_)
{
    const double alpha = as<double>(alpha_);
    const arma::Col<double> Am = as<arma::Col<double> >(A_);
    const arma::Col<double> Bm = as<arma::Col<double> >(B_);
    
    arma::Col<double> Cm = cpp_arma_vienna_vec_axpy(alpha, Am, Bm);
    
    return wrap(Cm);
}

//[[Rcpp::export]]
SEXP cpp_vienna_gpuVector_saxpy(SEXP alpha_, SEXP A_, SEXP B_)
{
    const float alpha = as<float>(alpha_);
    const arma::Col<float> Am = as<arma::Col<float> >(A_);
    const arma::Col<float> Bm = as<arma::Col<float> >(B_);
    
    arma::Col<float> Cm = cpp_arma_vienna_vec_axpy(alpha, Am, Bm);
    
    return wrap(Cm);
}
