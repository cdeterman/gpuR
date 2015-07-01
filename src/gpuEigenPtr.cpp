
#include <RcppEigen.h>

#include "eigen_helpers.hpp"

using namespace Rcpp;

// [[Rcpp::export]]
SEXP matrixToIntXptr(SEXP data)
{
    SEXP pMat = sexpToXptr<int>(data);
    return(pMat);
}

// [[Rcpp::export]]
SEXP matrixToFloatXptr(SEXP data)
{
    SEXP pMat = sexpToXptr<float>(data);
    return(pMat);
}


// [[Rcpp::export]]
SEXP matrixToDoubleXptr(SEXP data)
{
    SEXP pMat = sexpToXptr<double>(data);
    return(pMat);
}


// [[Rcpp::export]]
SEXP dXptrToSEXP(SEXP ptrA)
{
    typename MapMat<double>::Type A = XPtrToSEXP<double>(ptrA);
    return wrap(A);
}


// [[Rcpp::export]]
SEXP fXptrToSEXP(SEXP ptrA)
{
    typename MapMat<float>::Type A = XPtrToSEXP<float>(ptrA);
    return wrap(A);
}


// [[Rcpp::export]]
SEXP iXptrToSEXP(SEXP ptrA)
{
    typename MapMat<int>::Type A = XPtrToSEXP<int>(ptrA);
    return wrap(A);
}


/*
 * Empty matrix initializers
 */

// [[Rcpp::export]]
SEXP emptyIntXptr(int nr, int nc)
{
    SEXP pMat = emptyXptr<int>(nr,nc);
    return(pMat);
}


// [[Rcpp::export]]
SEXP emptyFloatXptr(int nr, int nc)
{
    SEXP pMat = emptyXptr<float>(nr,nc);
    return(pMat);
}


// [[Rcpp::export]]
SEXP emptyDoubleXptr(int nr, int nc)
{
    SEXP pMat = emptyXptr<double>(nr,nc);
    return(pMat);
}

