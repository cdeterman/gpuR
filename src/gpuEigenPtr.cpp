
#include <RcppEigen.h>

#include "gpuR/eigen_helpers.hpp"

using namespace Rcpp;

/*** vector imports ***/
 
// [[Rcpp::export]]
SEXP vectorToIntXptr(SEXP data)
{
    SEXP pMat = sexpVecToXptr<int>(data);
    return(pMat);
}

// [[Rcpp::export]]
SEXP vectorToFloatXptr(SEXP data)
{
    SEXP pMat = sexpVecToXptr<float>(data);
    return(pMat);
}


// [[Rcpp::export]]
SEXP vectorToDoubleXptr(SEXP data)
{
    SEXP pMat = sexpVecToXptr<double>(data);
    return(pMat);
}



/*** matrix imports ***/

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

/*** Vector XPtr to SEXP ***/

// [[Rcpp::export]]
SEXP dXptrToVecSEXP(SEXP ptrA)
{
    MapVec<double> A = XPtrToVecSEXP<double>(ptrA);
    return wrap(A);
}


// [[Rcpp::export]]
SEXP fXptrToVecSEXP(SEXP ptrA)
{
    MapVec<float> A = XPtrToVecSEXP<float>(ptrA);
    return wrap(A);
}


// [[Rcpp::export]]
SEXP iXptrToVecSEXP(SEXP ptrA)
{
    MapVec<int> A = XPtrToVecSEXP<int>(ptrA);
    return wrap(A);
}

/*** Matrix XPtr to SEXP ***/

// [[Rcpp::export]]
SEXP dXptrToSEXP(SEXP ptrA)
{
    MapMat<double> A = XPtrToSEXP<double>(ptrA);
    return wrap(A);
}


// [[Rcpp::export]]
SEXP fXptrToSEXP(SEXP ptrA)
{
    MapMat<float> A = XPtrToSEXP<float>(ptrA);
    return wrap(A);
}


// [[Rcpp::export]]
SEXP iXptrToSEXP(SEXP ptrA)
{
    MapMat<int> A = XPtrToSEXP<int>(ptrA);
    return wrap(A);
}

/*** Empty vector initializers ***/

// [[Rcpp::export]]
SEXP emptyVecIntXptr(int size)
{
    SEXP pVec = emptyVecXptr<int>(size);
    return(pVec);
}


// [[Rcpp::export]]
SEXP emptyVecFloatXptr(int size)
{
    SEXP pVec = emptyVecXptr<float>(size);
    return(pVec);
}


// [[Rcpp::export]]
SEXP emptyVecDoubleXptr(int size)
{
    SEXP pVec = emptyVecXptr<double>(size);
    return(pVec);
}

/*** Empty matrix initializers ***/

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

