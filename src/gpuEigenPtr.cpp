
#include "gpuR/windows_check.hpp"
#include <RcppEigen.h>

#include "gpuR/eigen_helpers.hpp"

using namespace Rcpp;


//copy an existing gpuMatrix
template <typename T>
SEXP
cpp_deepcopy_gpuMatrix(SEXP ptrA_)
{
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > A(ptrA->data(), ptrA->rows(), ptrA->cols());
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> *C = new Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(ptrA->rows(), ptrA->cols());
    
    // assign pointer
    *C = A;
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > pMat(C);
    return pMat;
}

//copy an existing gpuVector
template <typename T>
SEXP
cpp_deepcopy_gpuVector(SEXP ptrA_)
{
    XPtr<dynEigenVec<T> > pA(ptrA_);
    Eigen::Matrix<T, Eigen::Dynamic, 1> A = pA->data();
    dynEigenVec<T> *vec = new dynEigenVec<T>(A);
    XPtr<dynEigenVec<T> > pVec(vec);
    return pVec;
}

template <typename T>
SEXP
sliceGPUvec(const SEXP ptrA, int start, int end)
{
    XPtr<dynEigenVec<T> > pA(ptrA);
    dynEigenVec<T> *vec = new dynEigenVec<T>();
    vec->setPtr(pA->getPtr());
    vec->setRange(start, end);
    vec->updateSize();
    
    XPtr<dynEigenVec<T> > pVec(vec);
    return pVec;
}


template <typename T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> >
get_gpu_slice_vec(const SEXP ptrA)
{
    XPtr<dynEigenVec<T> > pVec(ptrA);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > A = pVec->data();
    return A;
}

template <typename T>
SEXP
get_gpu_slice_length(const SEXP ptrA)
{
    XPtr<dynEigenVec<T> > pVec(ptrA);
    int A = pVec->length();
    return wrap(A);
}

template <typename T>
T
GetVecElement(const SEXP data, const int idx)
{    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > A = EigenVecXPtrToMapEigenVec<T>(data);
    return(A(idx-1));
}

template <typename T>
void
SetVecElement(const SEXP data, const int idx, SEXP value)
{    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > A = EigenVecXPtrToMapEigenVec<T>(data);
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


// [[Rcpp::export]]
SEXP
get_gpu_slice_length(SEXP ptrA, const int type_flag)
{    
    switch(type_flag) {
        case 4:
            return get_gpu_slice_length<int>(ptrA);
        case 6:
            return get_gpu_slice_length<float>(ptrA);
        case 8:
            return get_gpu_slice_length<double>(ptrA);
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
SEXP sexpVecToEigenVecXptr(SEXP ptrA, const int size, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return sexpVecToEigenVecXptr<int>(ptrA, size);
        case 6:
            return sexpVecToEigenVecXptr<float>(ptrA, size);
        case 8:
            return sexpVecToEigenVecXptr<double>(ptrA, size);
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}


// convert SEXP Vector to Eigen matrix
template <typename T>
SEXP sexpVecToEigenXptr(SEXP A, int nr, int nc)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> *eigen_mat = new Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(nr, nc);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> temp = as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A);
    temp.resize(nr, nc);
    *eigen_mat = temp;
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > pMat(eigen_mat);
    return pMat;
}

// [[Rcpp::export]]
SEXP
sexpVecToEigenXptr(SEXP ptrA, const int nr, const int nc, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return sexpVecToEigenXptr<int>(ptrA, nr, nc);
        case 6:
            return sexpVecToEigenXptr<float>(ptrA, nr, nc);
        case 8:
            return sexpVecToEigenXptr<double>(ptrA, nr, nc);
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}


/*** matrix imports ***/

// [[Rcpp::export]]
SEXP
sexpToEigenXptr(SEXP ptrA, 
    const int nr,
    const int nc, 
    const int type_flag)
{
    switch(type_flag) {
        case 4:
            return sexpToEigenXptr<int>(ptrA, nr, nc);
        case 6:
            return sexpToEigenXptr<float>(ptrA, nr, nc);
        case 8:
            return sexpToEigenXptr<double>(ptrA, nr, nc);
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

/*** Vector XPtr to SEXP ***/

// [[Rcpp::export]]
SEXP
EigenVecXPtrToMapEigenVec(SEXP ptrA, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return wrap(EigenVecXPtrToMapEigenVec<int>(ptrA));
        case 6:
            return wrap(EigenVecXPtrToMapEigenVec<float>(ptrA));
        case 8:
            return wrap(EigenVecXPtrToMapEigenVec<double>(ptrA));
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
            return wrap(EigenXPtrToMapEigen<int>(ptrA));
        case 6:
            return wrap(EigenXPtrToMapEigen<float>(ptrA));
        case 8:
            return wrap(EigenXPtrToMapEigen<double>(ptrA));
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

/*** Empty vector initializers ***/

// [[Rcpp::export]]
SEXP
emptyEigenVecXptr(const int size, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return emptyEigenVecXptr<int>(size);;
        case 6:
            return emptyEigenVecXptr<float>(size);
        case 8:
            return emptyEigenVecXptr<double>(size);
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}


/*** Empty matrix initializers ***/


// [[Rcpp::export]]
SEXP
emptyEigenXptr(const int nr, const int nc, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return emptyEigenXptr<int>(nr, nc);;
        case 6:
            return emptyEigenXptr<float>(nr, nc);
        case 8:
            return emptyEigenXptr<double>(nr, nc);
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}
