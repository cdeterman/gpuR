
#include <RcppEigen.h>

#include "gpuR/eigen_helpers.hpp"

using namespace Rcpp;

// [[Rcpp::export]]
SEXP 
dGetMatRow(const SEXP data, const int idx)
{    
    MapMat<double> A = XPtrToSEXP<double>(data);
    Eigen::VectorXd Am = A.row(idx-1);
    return(wrap(Am));
}

// [[Rcpp::export]]
SEXP 
fGetMatRow(const SEXP data, const int idx)
{    
    MapMat<float> A = XPtrToSEXP<float>(data);
    Eigen::VectorXf Am = A.row(idx-1);
    return(wrap(Am));
}

// [[Rcpp::export]]
SEXP 
iGetMatRow(const SEXP data, const int idx)
{    
    MapMat<int> A = XPtrToSEXP<int>(data);
    Eigen::VectorXi Am = A.row(idx-1);
    return(wrap(Am));
}

// [[Rcpp::export]]
SEXP 
dGetMatCol(const SEXP data, const int idx)
{        
    MapMat<double> A = XPtrToSEXP<double>(data);
    Eigen::VectorXd Am = A.col(idx-1);
    return(wrap(Am));
}

// [[Rcpp::export]]
SEXP 
fGetMatCol(const SEXP data, const int idx)
{    
    MapMat<float> A = XPtrToSEXP<float>(data);
    Eigen::VectorXf Am = A.col(idx-1);
    return(wrap(Am));
}

// [[Rcpp::export]]
SEXP 
iGetMatCol(const SEXP data, const int idx)
{    
    MapMat<int> A = XPtrToSEXP<int>(data);
    Eigen::VectorXi Am = A.col(idx-1);
    return(wrap(Am));
}

// [[Rcpp::export]]
SEXP 
dGetMatElement(const SEXP data, const int nr, const int nc)
{        
    MapMat<double> A = XPtrToSEXP<double>(data);
    double value = A(nr-1, nc-1);
    return(wrap(value));
}

// [[Rcpp::export]]
SEXP 
fGetMatElement(const SEXP data, const int nr, const int nc)
{        
    MapMat<float> A = XPtrToSEXP<float>(data);
    float value = A(nr-1, nc-1);
    return(wrap(value));
}

// [[Rcpp::export]]
SEXP 
iGetMatElement(const SEXP data, const int nr, const int nc)
{        
    MapMat<int> A = XPtrToSEXP<int>(data);
    int value = A(nr-1, nc-1);
    return(wrap(value));
}


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

// [[Rcpp::export]]
SEXP vectorToIntMatXptr(SEXP data, int nr, int nc)
{
    SEXP pMat = sexpVecToMatXptr<int>(data, nr, nc);
    return(pMat);
}

// [[Rcpp::export]]
SEXP vectorToFloatMatXptr(SEXP data, int nr, int nc)
{
    SEXP pMat = sexpVecToMatXptr<float>(data, nr, nc);
    return(pMat);
}

// [[Rcpp::export]]
SEXP vectorToDoubleMatXptr(SEXP data, int nr, int nc)
{
    SEXP pMat = sexpVecToMatXptr<double>(data, nr, nc);
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

