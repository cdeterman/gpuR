#include <RcppEigen.h>

#include "gpuR/vcl_helpers.hpp"

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::MatrixXi;
using Eigen::VectorXd;
using Eigen::VectorXf;
using Eigen::VectorXi;

using namespace Rcpp;


//copy an existing Xptr
template <typename T>
SEXP
cpp_deepcopy_vclMatrix(SEXP ptrA_)
{    
    Rcpp::XPtr<viennacl::matrix<T> > ptrA(ptrA_);
//    viennacl::matrix<T> vcl_A = *ptrA;
    
    viennacl::matrix<T> *vcl_B = new viennacl::matrix<T>(ptrA->size1(), ptrA->size2());
    *vcl_B = *ptrA;
    Rcpp::XPtr<viennacl::matrix<T> > pMat(vcl_B);
    return pMat;
}

// scalar initialized ViennaCL matrix
template <typename T>
SEXP 
cpp_scalar_vclMatrix(
    SEXP scalar_, 
    const int nr, 
    const int nc,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    const T scalar = as<T>(scalar_);
    
    viennacl::matrix<T> *vcl_A = new viennacl::matrix<T>(nr,nc);
    *vcl_A = viennacl::scalar_matrix<T>(nr,nc, scalar);
    
    Rcpp::XPtr<viennacl::matrix<T> > pMat(vcl_A);
    return pMat;
}

// convert SEXP Vector to ViennaCL vector
template <typename T>
SEXP 
sexpVecToVCL(SEXP A, const int device_flag)
{
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am;
    Am = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A);
    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.size();
    
    viennacl::vector<T> *vcl_A = new viennacl::vector<T>(M);
    
    viennacl::copy(Am, *vcl_A); 
    
    Rcpp::XPtr<viennacl::vector<T> > pMat(vcl_A);
    return pMat;
}

// convert SEXP Vector to ViennaCL matrix
template <typename T>
SEXP 
vectorToMatVCL(SEXP A, const int nr, const int nc, const int device_flag)
{
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am;
    Am = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A);
    Am.resize(nr, nc);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Amm(&Am(0), nr, nc);
    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    viennacl::matrix<T> *vcl_A = new viennacl::matrix<T>(nr, nc);
    
    viennacl::copy(Amm, *vcl_A); 
    
    Rcpp::XPtr<viennacl::matrix<T> > pMat(vcl_A);
    return pMat;
}

// convert XPtr ViennaCL Vector to Eigen vector
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> VCLtoVecSEXP(SEXP A)
{
    Rcpp::XPtr<viennacl::vector<T> > pA(A);
    
    int M = pA->size();
    
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am(M);
    
    viennacl::copy(*pA, Am); 
    
    return Am;
}


// empty ViennaCL Vector
template <typename T>
SEXP emptyVecVCL(int length, const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    viennacl::vector<T> *vcl_A = new viennacl::vector<T>(length);
    *vcl_A = viennacl::zero_vector<T>(length);
    
    Rcpp::XPtr<viennacl::vector<T> > pMat(vcl_A);
    return pMat;
}

/*** vclVector get elements ***/

// Get viennacl column elements
template <typename T>
T
vclVecGetElement(SEXP &data, const int &idx)
{
    Rcpp::XPtr<viennacl::vector<T> > pA(data);
    viennacl::vector<T> &A = *pA;
    return(A(idx-1));
}

/*** vclVector set elements ***/

// Get viennacl column elements
template <typename T>
void
vclVecSetElement(SEXP &data, SEXP newdata, const int &idx)
{
    Rcpp::XPtr<viennacl::vector<T> > pA(data);
    viennacl::vector<T> &A = *pA;
    A(idx-1) = as<T>(newdata);
}

/*** vclMatrix setting elements ***/

// update viennacl column elements
template <typename T>
void
vclSetCol(SEXP data, SEXP newdata, const int nc)
{
    Rcpp::XPtr<viennacl::matrix<T> > pA(data);
    viennacl::matrix<T> &A = *pA;
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am;
    Am = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(newdata);
    
    for(unsigned int i = 0; i < A.size1(); i++){
        A(i, nc-1) = Am(i);
    } 
}

// update viennacl row elements
template <typename T>
void
vclSetRow(SEXP data, SEXP newdata, const int nr)
{
    Rcpp::XPtr<viennacl::matrix<T> > pA(data);
    viennacl::matrix<T> &A = *pA;
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am;
    Am = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(newdata);
    
    for(unsigned int i = 0; i < A.size2(); i++){
        A(nr-1, i) = Am(i);
    } 
}

// update viennacl element
template <typename T>
void
vclSetElement(SEXP data, SEXP newdata, const int nr, const int nc)
{
    Rcpp::XPtr<viennacl::matrix<T> > pA(data);
    viennacl::matrix<T> &A = *pA;
    
    A(nr-1, nc-1) = as<T>(newdata);
}

/*** vclMatrix get elements ***/

// Get viennacl column elements
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
vclGetCol(SEXP &data, const int &nc)
{
    Rcpp::XPtr<viennacl::matrix<T> > pA(data);
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am;
    Am = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(pA->size1());
    
    viennacl::vector<T> vcl_A(pA->size1());
    vcl_A = viennacl::column(*pA, nc-1);
    
    copy(vcl_A, Am);
    return(Am);
}

// Get viennacl row elements
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
vclGetRow(SEXP &data, const int &nr)
{
    Rcpp::XPtr<viennacl::matrix<T> > pA(data);
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am;
    Am = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(pA->size2());
    
    viennacl::vector<T> vcl_A(pA->size2());
    vcl_A = viennacl::row(*pA, nr-1);
    
    copy(vcl_A, Am);
    return(Am);
}

// Get viennacl row elements
template <typename T>
T
vclGetElement(SEXP &data, const int &nr, const int &nc)
{
    T value;
    Rcpp::XPtr<viennacl::matrix<T> > pA(data);
    viennacl::matrix<T> &A = *pA;
    value = A(nr-1, nc-1);
    return(value);
}


/*** vclMatrix deepcopy ***/
// [[Rcpp::export]]
SEXP
cpp_deepcopy_vclMatrix(SEXP ptrA, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return cpp_deepcopy_vclMatrix<int>(ptrA);
        case 6:
            return cpp_deepcopy_vclMatrix<float>(ptrA);
        case 8:
            return cpp_deepcopy_vclMatrix<double>(ptrA);
        default:
            throw Rcpp::exception("unknown type detected for gpuVectorSlice object!");
    }
}

/*** matrix imports ***/

// [[Rcpp::export]]
SEXP
matrixToVCL(SEXP ptrA, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return sexpToVCL<int>(ptrA);
        case 6:
            return sexpToVCL<float>(ptrA);
        case 8:
            return sexpToVCL<double>(ptrA);
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


/*** Matrix exports ***/

// [[Rcpp::export]]
SEXP
VCLtoMatSEXP(SEXP ptrA, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return wrap(VCLtoSEXP<int>(ptrA));
        case 6:
            return wrap(VCLtoSEXP<float>(ptrA));
        case 8:
            return wrap(VCLtoSEXP<double>(ptrA));
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

/*** Matrix initializers ***/

// [[Rcpp::export]]
SEXP
cpp_zero_vclMatrix(const int nr, const int nc, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return cpp_zero_vclMatrix<int>(nr, nc);
        case 6:
            return cpp_zero_vclMatrix<float>(nr, nc);
        case 8:
            return cpp_zero_vclMatrix<double>(nr, nc);
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


// [[Rcpp::export]]
SEXP
cpp_scalar_vclMatrix(
    SEXP scalar, 
    const int nr, const int nc, 
    const int type_flag,
    const int device_flag)
{
    switch(type_flag) {
        case 4:
            return cpp_scalar_vclMatrix<int>(scalar, nr, nc, device_flag);
        case 6:
            return cpp_scalar_vclMatrix<float>(scalar, nr, nc, device_flag);
        case 8:
            return cpp_scalar_vclMatrix<double>(scalar, nr, nc, device_flag);
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

/*** matrix element updates ***/

// [[Rcpp::export]]
void
vclSetCol(SEXP ptrA, const int nc, SEXP newdata, const int type_flag)
{
    switch(type_flag) {
        case 4:
            vclSetCol<int>(ptrA, newdata, nc);
            return;
        case 6:
            vclSetCol<float>(ptrA, newdata, nc);
            return;
        case 8:
            vclSetCol<double>(ptrA, newdata, nc);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
void
vclSetRow(SEXP ptrA, const int nr, SEXP newdata, const int type_flag)
{
    switch(type_flag) {
        case 4:
            vclSetRow<int>(ptrA, newdata, nr);
            return;
        case 6:
            vclSetRow<float>(ptrA, newdata, nr);
            return;
        case 8:
            vclSetRow<double>(ptrA, newdata, nr);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
void
vclSetElement(SEXP ptrA, const int nr, const int nc, SEXP newdata, const int type_flag)
{
    switch(type_flag) {
        case 4:
            vclSetElement<int>(ptrA, newdata, nr, nc);
            return;
        case 6:
            vclSetElement<float>(ptrA, newdata, nr, nc);
            return;
        case 8:
            vclSetElement<double>(ptrA, newdata, nr, nc);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

/*** get matrix elements ***/

// [[Rcpp::export]]
SEXP
vclGetCol(SEXP ptrA, const int nc, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return wrap(vclGetCol<int>(ptrA, nc));
        case 6:
            return wrap(vclGetCol<float>(ptrA, nc));
        case 8:
            return wrap(vclGetCol<double>(ptrA, nc));
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
SEXP
vclGetRow(SEXP ptrA, const int nr, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return wrap(vclGetRow<int>(ptrA, nr));
        case 6:
            return wrap(vclGetRow<float>(ptrA, nr));
        case 8:
            return wrap(vclGetRow<double>(ptrA, nr));
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
SEXP
vclGetElement(SEXP ptrA, const int nr, const int nc, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return wrap(vclGetElement<int>(ptrA, nr, nc));
        case 6:
            return wrap(vclGetElement<float>(ptrA, nr, nc));
        case 8:
            return wrap(vclGetElement<double>(ptrA, nr, nc));
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

/*** get vector elements ***/

// [[Rcpp::export]]
SEXP
vclVecGetElement(SEXP ptrA, const int idx, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return wrap(vclVecGetElement<int>(ptrA, idx));
        case 6:
            return wrap(vclVecGetElement<float>(ptrA, idx));
        case 8:
            return wrap(vclVecGetElement<double>(ptrA, idx));
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
vclVecSetElement(SEXP ptrA, const int idx, SEXP newdata, const int type_flag)
{
    switch(type_flag) {
        case 4:
            vclVecSetElement<int>(ptrA, newdata, idx);
            return;
        case 6:
            vclVecSetElement<float>(ptrA, newdata, idx);
            return;
        case 8:
            vclVecSetElement<double>(ptrA, newdata, idx);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

/*** vector imports ***/

// [[Rcpp::export]]
SEXP
vectorToVCL(SEXP ptrA, const int type_flag, const int device_flag)
{
    switch(type_flag) {
        case 4:
            return sexpVecToVCL<int>(ptrA, device_flag);
        case 6:
            return sexpVecToVCL<float>(ptrA, device_flag);
        case 8:
            return sexpVecToVCL<double>(ptrA, device_flag);
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
SEXP
vectorToMatVCL(
    SEXP ptrA, 
    const int nr,
    const int nc,
    const int type_flag, 
    const int device_flag)
{
    switch(type_flag) {
        case 4:
            return vectorToMatVCL<int>(ptrA, nr, nc, device_flag);
        case 6:
            return vectorToMatVCL<float>(ptrA, nr, nc, device_flag);
        case 8:
            return vectorToMatVCL<double>(ptrA, nr, nc, device_flag);
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


/*** Vector exports ***/

// [[Rcpp::export]]
SEXP
VCLtoVecSEXP(SEXP ptrA, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return wrap(VCLtoVecSEXP<int>(ptrA));
        case 6:
            return wrap(VCLtoVecSEXP<float>(ptrA));
        case 8:
            return wrap(VCLtoVecSEXP<double>(ptrA));
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

/*** Empty vector initializers ***/

// [[Rcpp::export]]
SEXP
emptyVecVCL(int length, const int type_flag, const int device_flag)
{
    switch(type_flag) {
        case 4:
            return emptyVecVCL<int>(length, device_flag);
        case 6:
            return emptyVecVCL<float>(length, device_flag);
        case 8:
            return emptyVecVCL<double>(length, device_flag);
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

