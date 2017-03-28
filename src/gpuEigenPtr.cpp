
#include "gpuR/windows_check.hpp"

#include "gpuR/dynEigenMat.hpp"
#include "gpuR/dynEigenVec.hpp"

#include <RcppEigen.h>

using namespace Rcpp;


// convert SEXP Matrix to Eigen matrix
template <typename T>
SEXP 
getRmatEigenAddress(SEXP A, const int nr, const int nc)
{    
    dynEigenMat<T> *mat = new dynEigenMat<T>(A);
    Rcpp::XPtr<dynEigenMat<T> > pMat(mat);
    return pMat;
}

// convert SEXP Vector to Eigen Vector (i.e. 1 column matrix)
template <typename T>
SEXP 
sexpVecToEigenVecXptr(SEXP A, const int size)
{
    dynEigenVec<T> *vec = new dynEigenVec<T>(A);
    Rcpp::XPtr<dynEigenVec<T> > pVec(vec);
    return pVec;
}

// convert an XPtr back to a MapVec object to ultimately 
// be returned as a SEXP object
template <typename T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > 
getEigenMatrix(SEXP ptrA_)
{
    Rcpp::XPtr<dynEigenVec<T> > pVec(ptrA_);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > MapVec = pVec->data();
    return MapVec;
}

template <typename T>
void
SetMatRow(SEXP data, const int idx, SEXP value)
{    
    Rcpp::XPtr<dynEigenMat<T> > pMat(data);
    pMat->setRow(value, idx);
}


template <typename T>
void
SetMatCol(SEXP data, const int idx, SEXP value)
{    
    Rcpp::XPtr<dynEigenMat<T> > pMat(data);
    pMat->setCol(value, idx);
}

template <typename T>
void
SetMatElement(SEXP data, const int nr, const int nc, SEXP value)
{    
    Rcpp::XPtr<dynEigenMat<T> > pMat(data);
    pMat->setElement(value, nr, nc);
}

template <typename T>
SEXP
GetMatRow(const SEXP data, const int idx)
{    
    Rcpp::XPtr<dynEigenMat<T> > pMat(data);
    return(wrap(pMat->getRow(idx)));
}

template <typename T>
SEXP
GetMatCol(const SEXP data, const int idx)
{    
    Rcpp::XPtr<dynEigenMat<T> > pMat(data);
    return(wrap(pMat->getCol(idx)));
}

template <typename T>
SEXP
GetMatElement(const SEXP data, const int nr, const int nc)
{    
    Rcpp::XPtr<dynEigenMat<T> > pMat(data);
    return(wrap(pMat->getElement(nr, nc)));
}

// create an empty eigen matrix
template <typename T>
SEXP emptyEigenXptr(int nr, int nc)
{
    dynEigenMat<T> *mat = new dynEigenMat<T>(nr, nc);
    //    std::cout << mat->data() << std::endl;
    Rcpp::XPtr<dynEigenMat<T> > pMat(mat);
    return pMat;
}

// create an empty eigen vector
template <typename T>
SEXP 
emptyEigenVecXptr(const int size)
{    
    dynEigenVec<T> *vec = new dynEigenVec<T>(size);
    Rcpp::XPtr<dynEigenVec<T> > pVec(vec);
    return pVec;
}

template <typename T>
void
setCols(SEXP ptrA_, StringVector names){
    
    Rcpp::XPtr<dynEigenMat<T> > ptrA(ptrA_);
    ptrA->setColumnNames(names);
    
    return;
}

template <typename T>
StringVector
getCols(SEXP ptrA_){
    
    Rcpp::XPtr<dynEigenMat<T> > ptrA(ptrA_);
    StringVector cnames = ptrA->getColumnNames();
    
    return cnames;
}

// get diagonal of gpuMatrix
template <typename T>
void
cpp_gpuMatrix_get_diag(SEXP ptrA_, SEXP ptrB_)
{
    Rcpp::XPtr<dynEigenMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<T> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::OuterStride<> > pA = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> >pB = ptrB->data();
    
    pB = pA.diagonal();
}

// set diagonal with gpuVector
template <typename T>
void
cpp_gpuMat_gpuVec_set_diag(SEXP ptrA_, SEXP ptrB_)
{
    Rcpp::XPtr<dynEigenMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<T> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::OuterStride<> > pA = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> >pB = ptrB->data();
    
    pA.diagonal() = pB;
}


//copy an existing gpuMatrix
template <typename T>
SEXP
cpp_deepcopy_gpuMatrix(SEXP ptrA_)
{
//    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
//    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > A(ptrA->data(), ptrA->rows(), ptrA->cols());
//    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> *C = new Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(ptrA->rows(), ptrA->cols());
//    
//    // assign pointer
//    *C = A;
//    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > pMat(C);
//    return pMat;
    
    XPtr<dynEigenMat<T> > pA(ptrA_);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A = pA->data();
    dynEigenMat<T> *mat = new dynEigenMat<T>(A);
    XPtr<dynEigenMat<T> > pMat(mat);
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
SEXP
gpuMatBlock(
    const SEXP ptrA, 
    int rowStart, int rowEnd,
    int colStart, int colEnd)
{
    XPtr<dynEigenMat<T> > pA(ptrA);
    dynEigenMat<T> *mat = new dynEigenMat<T>();
    mat->setHostPtr(pA->getHostPtr());
    mat->setRange(rowStart, rowEnd, colStart, colEnd);
    mat->setSourceDim(pA->nrow(), pA->ncol());
    mat->updateDim();
    
    XPtr<dynEigenMat<T> > pMat(mat);
    return pMat;
}

//cbind gpuMatrix objects
template <typename T>
SEXP
cpp_cbind_gpuMatrix(SEXP ptrA_, SEXP ptrB_)
{    
    XPtr<dynEigenMat<T> > pA(ptrA_);
    XPtr<dynEigenMat<T> > pB(ptrB_);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A = pA->data();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> B = pB->data();
    
    // initialize new matrix
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> C(A.rows(), A.cols() + B.cols());
    C << A,B;
    
    dynEigenMat<T> *mat = new dynEigenMat<T>(C);
    XPtr<dynEigenMat<T> > pMat(mat);
    return pMat;
}

//rbind gpuMatrix objects
template <typename T>
SEXP
cpp_rbind_gpuMatrix(SEXP ptrA_, SEXP ptrB_)
{    
    XPtr<dynEigenMat<T> > pA(ptrA_);
    XPtr<dynEigenMat<T> > pB(ptrB_);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A = pA->data();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> B = pB->data();
    
    // initialize new matrix
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> C(A.rows()+B.rows(), A.cols());
    C << A,B;
    
    dynEigenMat<T> *mat = new dynEigenMat<T>(C);
    XPtr<dynEigenMat<T> > pMat(mat);
    return pMat;
}


// template <typename T>
// Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> >
// get_gpu_slice_vec(const SEXP ptrA)
// {
//     XPtr<dynEigenVec<T> > pVec(ptrA);
//     Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > A = pVec->data();
//     return A;
// }
// 
// template <typename T>
// SEXP
// get_gpu_slice_length(const SEXP ptrA)
// {
//     XPtr<dynEigenVec<T> > pVec(ptrA);
//     int A = pVec->length();
//     return wrap(A);
// }

template <typename T>
T
GetVecElement(const SEXP data, const int idx)
{    
    XPtr<dynEigenVec<T> > pVec(data);
    return(pVec->getElement(idx));
}

template <typename T>
void
SetVecElement(const SEXP data, const int idx, SEXP value)
{    
    XPtr<dynEigenVec<T> > pVec(data);
    pVec->setElement(idx, value);
    return;
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
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

/*** gpuMatrix cbind ***/
// [[Rcpp::export]]
SEXP
cpp_cbind_gpuMatrix(SEXP ptrA, SEXP ptrB, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return cpp_cbind_gpuMatrix<int>(ptrA, ptrB);
        case 6:
            return cpp_cbind_gpuMatrix<float>(ptrA, ptrB);
        case 8:
            return cpp_cbind_gpuMatrix<double>(ptrA, ptrB);
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

/*** gpuMatrix rbind ***/
// [[Rcpp::export]]
SEXP
cpp_rbind_gpuMatrix(SEXP ptrA, SEXP ptrB, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return cpp_rbind_gpuMatrix<int>(ptrA, ptrB);
        case 6:
            return cpp_rbind_gpuMatrix<float>(ptrA, ptrB);
        case 8:
            return cpp_rbind_gpuMatrix<double>(ptrA, ptrB);
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
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
            throw Rcpp::exception("unknown type detected for gpuVector object!");
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
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}
    
    
/*** Matrix Block ***/
// [[Rcpp::export]]
SEXP
gpuMatBlock(
    SEXP ptrA, 
    int rowStart, int rowEnd,
    int colStart, int colEnd,
    const int type_flag)
{    
    switch(type_flag) {
        case 4:
            return gpuMatBlock<int>(ptrA, rowStart, rowEnd, colStart, colEnd);
        case 6:
            return gpuMatBlock<float>(ptrA, rowStart, rowEnd, colStart, colEnd);
        case 8:
            return gpuMatBlock<double>(ptrA, rowStart, rowEnd, colStart, colEnd);
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}


// // [[Rcpp::export]]
// SEXP
// get_gpu_slice_vec(SEXP ptrA, const int type_flag)
// {    
//     switch(type_flag) {
//         case 4:
//             return wrap(get_gpu_slice_vec<int>(ptrA));
//         case 6:
//             return wrap(get_gpu_slice_vec<float>(ptrA));
//         case 8:
//             return wrap(get_gpu_slice_vec<double>(ptrA));
//         default:
//             throw Rcpp::exception("unknown type detected for gpuVectorSlice object!");
//     }
// }
// 
// 
// // [[Rcpp::export]]
// SEXP
// get_gpu_slice_length(SEXP ptrA, const int type_flag)
// {    
//     switch(type_flag) {
//         case 4:
//             return get_gpu_slice_length<int>(ptrA);
//         case 6:
//             return get_gpu_slice_length<float>(ptrA);
//         case 8:
//             return get_gpu_slice_length<double>(ptrA);
//         default:
//             throw Rcpp::exception("unknown type detected for gpuVectorSlice object!");
//     }
// }

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
//    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> *eigen_mat = new Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(nr, nc);
//    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> temp = as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A);
//    temp.resize(nr, nc);
//    *eigen_mat = temp;
//    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > pMat(eigen_mat);
//    return pMat;
    
    dynEigenMat<T> *mat = new dynEigenMat<T>(nr, nc);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> temp = as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A);
    temp.resize(nr, nc);
    mat->setMatrix(temp);
    XPtr<dynEigenMat<T> > pMat(mat);
    return pMat;
}

// convert SEXP Vector to Eigen matrix
template <typename T>
SEXP initScalarEigenXptr(T A, int nr, int nc)
{    
    dynEigenMat<T> *mat = new dynEigenMat<T>(A, nr, nc);
    XPtr<dynEigenMat<T> > pMat(mat);
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


// [[Rcpp::export]]
SEXP
initScalarEigenXptr(SEXP scalar, const int nr, const int nc, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return initScalarEigenXptr<int>(as<int>(scalar), nr, nc);
        case 6:
            return initScalarEigenXptr<float>(as<float>(scalar), nr, nc);
        case 8:
            return initScalarEigenXptr<double>(as<double>(scalar), nr, nc);
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

/*** matrix imports ***/

// [[Rcpp::export]]
SEXP
getRmatEigenAddress(SEXP ptrA, 
    const int nr,
    const int nc, 
    const int type_flag)
{
    switch(type_flag) {
        case 4:
            return getRmatEigenAddress<int>(ptrA, nr, nc);
        case 6:
            return getRmatEigenAddress<float>(ptrA, nr, nc);
        case 8:
            return getRmatEigenAddress<double>(ptrA, nr, nc);
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

/*** Vector XPtr to SEXP ***/

// [[Rcpp::export]]
SEXP
getEigenMatrix(SEXP ptrA, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return wrap(getEigenMatrix<int>(ptrA));
        case 6:
            return wrap(getEigenMatrix<float>(ptrA));
        case 8:
            return wrap(getEigenMatrix<double>(ptrA));
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}


template <typename T>
SEXP
MatXptrToMatSEXP(SEXP ptrA){
    Rcpp::XPtr<dynEigenMat<T> > pMat(ptrA);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::OuterStride<> > mapA = pMat->data();
    return wrap(mapA);
}

/*** Matrix XPtr to SEXP ***/

// [[Rcpp::export]]
SEXP
MatXptrToMatSEXP(SEXP ptrA, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return(MatXptrToMatSEXP<int>(ptrA));
        case 6:
            return(MatXptrToMatSEXP<float>(ptrA));
        case 8:
            return(MatXptrToMatSEXP<double>(ptrA));
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


// [[Rcpp::export]]
void
setCols(SEXP ptrA, StringVector names, const int type_flag)
{
    switch(type_flag) {
    case 4:
        setCols<int>(ptrA, names);
        return;
    case 6:
        setCols<float>(ptrA, names);
        return;
    case 8:
        setCols<double>(ptrA, names);
        return;
    default:
        throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}


// [[Rcpp::export]]
StringVector
getCols(SEXP ptrA, const int type_flag)
{
    switch(type_flag) {
    case 4:
        return getCols<int>(ptrA);
    case 6:
        return getCols<float>(ptrA);
    case 8:
        return getCols<double>(ptrA);
    default:
        throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

/*** gpuMatrix diag ***/
// [[Rcpp::export]]
void
cpp_gpuMatrix_get_diag(SEXP ptrA, SEXP ptrB, const int type_flag)
{
    switch(type_flag) {
    case 4:
        cpp_gpuMatrix_get_diag<int>(ptrA, ptrB);
        return;
    case 6:
        cpp_gpuMatrix_get_diag<float>(ptrA, ptrB);
        return;
    case 8:
        cpp_gpuMatrix_get_diag<double>(ptrA, ptrB);
        return;
    default:
        throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// gpuMatrix set diag with gpuVector
// [[Rcpp::export]]
void
cpp_gpuMat_gpuVec_set_diag(SEXP ptrA, SEXP ptrB, const int type_flag)
{
    switch(type_flag) {
    case 4:
        cpp_gpuMat_gpuVec_set_diag<int>(ptrA, ptrB);
        return;
    case 6:
        cpp_gpuMat_gpuVec_set_diag<float>(ptrA, ptrB);
        return;
    case 8:
        cpp_gpuMat_gpuVec_set_diag<double>(ptrA, ptrB);
        return;
    default:
        throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

