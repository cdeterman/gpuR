
#include <RcppEigen.h>

#include "gpuR/eigen_helpers.hpp"

using namespace Rcpp;

template <typename T>
T
GetVecElement(const SEXP data, const int idx)
{    
    MapVec<T> A = XPtrToVecSEXP<T>(data);
    return(A(idx-1));
}

template <typename T>
void
SetVecElement(const SEXP data, const int idx, SEXP value)
{    
    MapVec<T> A = XPtrToVecSEXP<T>(data);
    A(idx-1) = as<T>(value);
}

/*** Get/Set Vector Elements ***/

// [[Rcpp::export]]
SEXP
GetVecElement(SEXP ptrA, const int idx, const int type_flag)
{    
    switch(type_flag) {
        case 4:
            return wrap(GetVecElement<int>(ptrA, idx));
        case 6:
            return wrap(GetVecElement<float>(ptrA, idx));
        case 8:
            return wrap(GetVecElement<double>(ptrA, idx));
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
SetVecElement(SEXP ptrA, const int idx, SEXP value, const int type_flag)
{    
    switch(type_flag) {
        case 4:
            SetVecElement<int>(ptrA, idx, value);
            return;
        case 6:
            SetVecElement<float>(ptrA, idx, value);
            return;
        case 8:
            SetVecElement<double>(ptrA, idx, value);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

/*** Get/Set Matrix Elements ***/

// [[Rcpp::export]]
void
SetMatRow(SEXP ptrA, const int idx, SEXP value, const int type_flag)
{    
    switch(type_flag) {
        case 4:
            SetMatRow<int>(ptrA, idx, value);
            return;
        case 6:
            SetMatRow<float>(ptrA, idx, value);
            return;
        case 8:
            SetMatRow<double>(ptrA, idx, value);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
SetMatCol(SEXP ptrA, const int idx, SEXP value, const int type_flag)
{    
    switch(type_flag) {
        case 4:
            SetMatCol<int>(ptrA, idx, value);
            return;
        case 6:
            SetMatCol<float>(ptrA, idx, value);
            return;
        case 8:
            SetMatCol<double>(ptrA, idx, value);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
SetMatElement(SEXP ptrA, const int nr, const int nc, SEXP value, const int type_flag)
{    
    switch(type_flag) {
        case 4:
            SetMatElement<int>(ptrA, nr, nc, value);
            return;
        case 6:
            SetMatElement<float>(ptrA, nr, nc, value);
            return;
        case 8:
            SetMatElement<double>(ptrA, nr, nc, value);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
SEXP
GetMatRow(SEXP ptrA, const int idx, const int type_flag)
{    
    switch(type_flag) {
        case 4:
            return GetMatRow<int>(ptrA, idx);
        case 6:
            return GetMatRow<float>(ptrA, idx);
        case 8:
            return GetMatRow<double>(ptrA, idx);
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
SEXP
GetMatCol(SEXP ptrA, const int idx, const int type_flag)
{    
    switch(type_flag) {
        case 4:
            return GetMatCol<int>(ptrA, idx);
        case 6:
            return GetMatCol<float>(ptrA, idx);
        case 8:
            return GetMatCol<double>(ptrA, idx);
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
SEXP
GetMatElement(SEXP ptrA, const int nr, const int nc, const int type_flag)
{    
    switch(type_flag) {
        case 4:
            return GetMatElement<int>(ptrA, nr, nc);
        case 6:
            return GetMatElement<float>(ptrA, nr, nc);
        case 8:
            return GetMatElement<double>(ptrA, nr, nc);
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}


/*** vector imports ***/

// [[Rcpp::export]]
SEXP vectorToSEXP(SEXP ptrA, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return sexpVecToXptr<int>(ptrA);
        case 6:
            return sexpVecToXptr<float>(ptrA);
        case 8:
            return sexpVecToXptr<double>(ptrA);
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
SEXP
vectorToMat(SEXP ptrA, const int nr, const int nc, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return sexpVecToMatXptr<int>(ptrA, nr, nc);
        case 6:
            return sexpVecToMatXptr<float>(ptrA, nr, nc);
        case 8:
            return sexpVecToMatXptr<double>(ptrA, nr, nc);
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}


/*** matrix imports ***/

// [[Rcpp::export]]
SEXP
matrixToGPUXptr(SEXP ptrA, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return sexpToXptr<int>(ptrA);
        case 6:
            return sexpToXptr<float>(ptrA);
        case 8:
            return sexpToXptr<double>(ptrA);
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

/*** Vector XPtr to SEXP ***/

// [[Rcpp::export]]
SEXP
VecXptrToVecSEXP(SEXP ptrA, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return wrap(XPtrToVecSEXP<int>(ptrA));
        case 6:
            return wrap(XPtrToVecSEXP<float>(ptrA));
        case 8:
            return wrap(XPtrToVecSEXP<double>(ptrA));
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

/*** Matrix XPtr to SEXP ***/

// [[Rcpp::export]]
SEXP
MatXptrToMatSEXP(SEXP ptrA, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return wrap(XPtrToSEXP<int>(ptrA));
        case 6:
            return wrap(XPtrToSEXP<float>(ptrA));
        case 8:
            return wrap(XPtrToSEXP<double>(ptrA));
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

/*** Empty vector initializers ***/

// [[Rcpp::export]]
SEXP
emptyVecXptr(const int size, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return emptyVecXptr<int>(size);;
        case 6:
            return emptyVecXptr<float>(size);
        case 8:
            return emptyVecXptr<double>(size);
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}


/*** Empty matrix initializers ***/


// [[Rcpp::export]]
SEXP
emptyMatXptr(const int nr, const int nc, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return emptyXptr<int>(nr, nc);;
        case 6:
            return emptyXptr<float>(nr, nc);
        case 8:
            return emptyXptr<double>(nr, nc);
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}
