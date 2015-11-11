

#include "gpuR/windows_check.hpp"

// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/dynEigen.hpp"

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

// ViennaCL headers
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"

using namespace Rcpp;

/*** gpuMatrix templates ***/

template <typename T>
void 
cpp_gpuMatrix_gemm(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<dynEigen<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<T> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());    
    
    const int M = Am.cols();
    const int K = Am.rows();
    const int N = Bm.rows();
    const int P = Bm.cols();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(N,P);
    viennacl::matrix<T> vcl_C(K,P);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_C = viennacl::linalg::prod(vcl_A, vcl_B);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuMatrix_crossprod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<dynEigen<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<T> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    const int M = Am.cols();
    const int K = Am.rows();
    const int N = Bm.rows();
    const int P = Bm.cols();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(N,P);
    viennacl::matrix<T> vcl_C(Cm.rows(),Cm.cols());
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_C = viennacl::linalg::prod(trans(vcl_A), vcl_B);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuMatrix_tcrossprod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<dynEigen<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<T> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    const int M = Am.cols();
    const int K = Am.rows();
    const int N = Bm.rows();
    const int P = Bm.cols();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(N,P);
    viennacl::matrix<T> vcl_C(K,N);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_C = viennacl::linalg::prod(vcl_A, trans(vcl_B));
    
    viennacl::copy(vcl_C, Cm);
}


/*** gpuMatrix Functions ***/

// [[Rcpp::export]]
void
cpp_gpuMatrix_gemm(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_gemm<int>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_gemm<float>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_gemm<double>(ptrA, ptrB, ptrC, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_crossprod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_crossprod<int>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_crossprod<float>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_crossprod<double>(ptrA, ptrB, ptrC, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_gpuMatrix_tcrossprod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_tcrossprod<int>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_tcrossprod<float>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_tcrossprod<double>(ptrA, ptrB, ptrC, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


/*** vclMatrix Templates ***/

template <typename T>
void cpp_vclMatrix_gemm(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<T> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<T> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::prod(*ptrA, *ptrB);
}

template <typename T>
void 
cpp_vclMatrix_crossprod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<T> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<T> > ptrC(ptrC_);
    
    *ptrC = viennacl::linalg::prod(trans(*ptrA), *ptrB);
}

template <typename T>
void
cpp_vclMatrix_tcrossprod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<T> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<T> > ptrC(ptrC_);
    
    *ptrC = viennacl::linalg::prod(*ptrA, trans(*ptrB));
}

/*** vclMatrix Functions ***/

// [[Rcpp::export]]
void
cpp_vclMatrix_gemm(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclMatrix_gemm<int>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 6:
            cpp_vclMatrix_gemm<float>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 8:
            cpp_vclMatrix_gemm<double>(ptrA, ptrB, ptrC, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclMatrix_crossprod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclMatrix_crossprod<int>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 6:
            cpp_vclMatrix_crossprod<float>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 8:
            cpp_vclMatrix_crossprod<double>(ptrA, ptrB, ptrC, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclMatrix_tcrossprod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclMatrix_tcrossprod<int>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 6:
            cpp_vclMatrix_tcrossprod<float>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 8:
            cpp_vclMatrix_tcrossprod<double>(ptrA, ptrB, ptrC, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}



