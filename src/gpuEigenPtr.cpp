
#include "gpuR/windows_check.hpp"
#include <RcppEigen.h>

#include "gpuR/eigen_helpers.hpp"

using namespace Rcpp;


//copy an existing gpuMatrix
template <typename T>
SEXP
cpp_deepcopy_gpuMatrix(SEXP ptrA_)
{
    Rcpp::XPtr<dynEigen<T> > ptrA(ptrA_);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A(ptrA->data());
    dynEigen<T> *C = new dynEigen<T>(A);
    Rcpp::XPtr<dynEigen<T> > pMat(C);
    return pMat;
}

//copy an existing gpuVector
template <typename T>
SEXP
cpp_deepcopy_gpuVector(SEXP ptrA_)
{
    Rcpp::XPtr<dynEigenVec<T> > ptrA(ptrA_);
    Eigen::Matrix<T, Eigen::Dynamic, 1> A(ptrA->data());
    dynEigenVec<T> *C = new dynEigenVec<T>(A);
    Rcpp::XPtr<dynEigenVec<T> > pMat(C);
    return pMat;
}

//// convert SEXP Vector to Eigen matrix
//template <typename T>
//SEXP sexpVecToMatXptr(SEXP A, int nr, int nc)
//{
//    dynEigen<T> *C = new dynEigen<T>(A, nr, nc);
//    Rcpp::XPtr<dynEigen<T> > pMat(C);
//    return pMat;
//}
//
//// convert an XPtr back to a MapMat object to ultimately 
//// be returned as a SEXP object
//template <typename T>
//Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > XPtrToSEXP(SEXP ptrA_)
//{
//    Rcpp::XPtr<dynEigen<T> > ptrA(ptrA_);
//    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > A(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
//    return A;
//}

template <typename T>
SEXP
sliceGPUvec(const SEXP ptrA, int start, int end)
{
//    Rcpp::XPtr<dynEigenVec<T> > pVec(ptrA);
//    dynEigenVec<T> &vec = *pVec;
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > A = XPtrToVecSEXP<T>(ptrA);
    dynEigenVec<T> *C = new dynEigenVec<T>(A, start, end);
//    Eigen::VectorBlock<Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> >, Eigen::Dynamic> temp = A.segment(start, end);
//    Eigen::VectorBlock<T> slicedVec = A.segment(start, end);
//    Rcpp::XPtr<dynEigenVec<T>> slicedVecXPtr = sexpVecToXptr(temp);
//    return(slicedVecXPtr);
    Rcpp::XPtr<dynEigenVec<T> > pVec(C);
    return pVec;
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
get_gpu_slice_vec(const SEXP ptrA)
{
     Rcpp::XPtr<dynEigenVec<T> > pVec(ptrA);
     Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > A(pVec->ptr(), pVec->length());
     
     Eigen::Matrix<T, Eigen::Dynamic, 1> vec = A.segment(pVec->start(), pVec->end());
     return vec;
}

template <typename T>
T
GetVecElement(const SEXP data, const int idx)
{    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > A = XPtrToVecSEXP<T>(data);
    return(A(idx-1));
}

template <typename T>
void
SetVecElement(const SEXP data, const int idx, SEXP value)
{    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > A = XPtrToVecSEXP<T>(data);
    A(idx-1) = as<T>(value);
}


/*** gpuMatrix deepcopy ***/
// [[Rcpp::export]]
SEXP
cpp_deepcopy_gpuMatrix(SEXP ptrA, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return cpp_deepcopy_gpuMatrix<int>(ptrA);
        case 6:
            return cpp_deepcopy_gpuMatrix<float>(ptrA);
        case 8:
            return cpp_deepcopy_gpuMatrix<double>(ptrA);
        default:
            throw Rcpp::exception("unknown type detected for gpuVectorSlice object!");
    }
}

/*** gpuVector deepcopy ***/
// [[Rcpp::export]]
SEXP
cpp_deepcopy_gpuVector(SEXP ptrA, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return cpp_deepcopy_gpuVector<int>(ptrA);
        case 6:
            return cpp_deepcopy_gpuVector<float>(ptrA);
        case 8:
            return cpp_deepcopy_gpuVector<double>(ptrA);
        default:
            throw Rcpp::exception("unknown type detected for gpuVectorSlice object!");
    }
}

/*** Slice Vector ***/
// [[Rcpp::export]]
SEXP
sliceGPUvec(SEXP ptrA, const int start, const int end, const int type_flag)
{    
    switch(type_flag) {
        case 4:
            return sliceGPUvec<int>(ptrA, start, end);
        case 6:
            return sliceGPUvec<float>(ptrA, start, end);
        case 8:
            return sliceGPUvec<double>(ptrA, start, end);
        default:
            throw Rcpp::exception("unknown type detected for gpuVectorSlice object!");
    }
}

// [[Rcpp::export]]
SEXP
get_gpu_slice_vec(SEXP ptrA, const int type_flag)
{    
    switch(type_flag) {
        case 4:
            return wrap(get_gpu_slice_vec<int>(ptrA));
        case 6:
            return wrap(get_gpu_slice_vec<float>(ptrA));
        case 8:
            return wrap(get_gpu_slice_vec<double>(ptrA));
        default:
            throw Rcpp::exception("unknown type detected for gpuVectorSlice object!");
    }
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
