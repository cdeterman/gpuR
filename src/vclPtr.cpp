#include <RcppEigen.h>

#include "gpuR/vcl_helpers.hpp"


using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::MatrixXi;

using namespace Rcpp;

/*** matrix imports ***/

// [[Rcpp::export]]
SEXP matrixToIntVCL(SEXP data)
{
    SEXP pMat = sexpToVCL<int>(data);
    return(pMat);
}

// [[Rcpp::export]]
SEXP matrixToFloatVCL(SEXP data)
{
    SEXP pMat = sexpToVCL<float>(data);
    return(pMat);
}

// [[Rcpp::export]]
SEXP matrixToDoubleVCL(SEXP data)
{
    SEXP pMat = sexpToVCL<double>(data);
    return(pMat);
}


/*** Matrix exports ***/

// [[Rcpp::export]]
SEXP dVCLtoSEXP(SEXP ptrA)
{
    MatrixXd A = VCLtoSEXP<double>(ptrA);
    return wrap(A);
}


// [[Rcpp::export]]
SEXP fVCLtoSEXP(SEXP ptrA)
{
    MatrixXf A = VCLtoSEXP<float>(ptrA);
    return wrap(A);
}


// [[Rcpp::export]]
SEXP iVCLtoSEXP(SEXP ptrA)
{
    MatrixXi A = VCLtoSEXP<int>(ptrA);
    return wrap(A);
}

/*** Empty matrix initializers ***/

// [[Rcpp::export]]
SEXP emptyIntVCL(int nr, int nc)
{
    SEXP pMat = emptyVCL<int>(nr,nc);
    return(pMat);
}


// [[Rcpp::export]]
SEXP emptyFloatVCL(int nr, int nc)
{
    SEXP pMat = emptyVCL<float>(nr,nc);
    return(pMat);
}


// [[Rcpp::export]]
SEXP emptyDoubleVCL(int nr, int nc)
{
    SEXP pMat = emptyVCL<double>(nr,nc);
    return(pMat);
}
