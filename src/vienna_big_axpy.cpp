
// Armadillo headers (disable BLAS and LAPACK to avoid linking issues)
#define ARMA_DONT_USE_BLAS
#define ARMA_DONT_USE_LAPACK

// armadillo headers for handling the R input data
#include <RcppArmadillo.h>

// big.matrix accession headers
#include <bigmemory/MatrixAccessor.hpp>

// Armadillo helpers for big.matrix objects
#include "arma_helpers.hpp"

// ViennaCL headers
#include "vcl_big_axpy.hpp"

using namespace Rcpp;

//[[Rcpp::export]]
void cpp_vienna_gpuBigMatrix_daxpy(SEXP alpha_, SEXP A_, SEXP B_)
{
    const double alpha = as<double>(alpha_);
    const arma::Mat<double> Am = ConvertBMtoArma<double>(A_);
    arma::Mat<double> Bm = ConvertBMtoArma<double>(B_);
    
    cpp_arma_vienna_big_axpy(alpha, Am, Bm);
}

//[[Rcpp::export]]
void cpp_vienna_gpuBigMatrix_saxpy(SEXP alpha_, SEXP A_, SEXP B_)
{
    const float alpha = as<float>(alpha_);
    const arma::Mat<float> Am = ConvertBMtoArma<float>(A_);
    arma::Mat<float> Bm = ConvertBMtoArma<float>(B_);
    
    cpp_arma_vienna_big_axpy(alpha, Am, Bm);
}
