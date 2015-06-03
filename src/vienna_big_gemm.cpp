
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
#include "vcl_big_gemm.hpp"

using namespace Rcpp;

//[[Rcpp::export]]
SEXP cpp_vienna_gpuBigMatrix_dgemm(SEXP A_, SEXP B_, SEXP C_)
{
    const arma::Mat<double> Am = ConvertBMtoArma<double>(A_);
    const arma::Mat<double> Bm = ConvertBMtoArma<double>(B_);
    arma::Mat<double> Cm = ConvertBMtoArma<double>(C_);
    
    cpp_arma_vienna_big_gemm(Am, Bm, Cm);
    
    return wrap(Cm);
}

//[[Rcpp::export]]
SEXP cpp_vienna_gpuBigMatrix_sgemm(SEXP A_, SEXP B_, SEXP C_)
{
    const arma::Mat<float> Am = ConvertBMtoArma<float>(A_);
    const arma::Mat<float> Bm = ConvertBMtoArma<float>(B_);
    arma::Mat<float> Cm = ConvertBMtoArma<float>(C_);
    
    cpp_arma_vienna_big_gemm(Am, Bm, Cm);
    
    return wrap(Cm);
}
