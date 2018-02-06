

#include "gpuR/windows_check.hpp"

// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/dynEigenMat.hpp"
#include "gpuR/dynVCLMat.hpp"
#include "gpuR/dynVCLVec.hpp"

// ViennaCL headers
#ifndef BACKEND_CUDA
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#endif
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"

using namespace Rcpp;

/*** gpuMatrix templates ***/

template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else
    void
#endif 
cpp_gpuMatrix_gemm(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_)
{    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    XPtr<dynEigenMat<T> > ptrC(ptrC_);
    
    const int M = ptrC->row_end() - ptrC->row_start() + 1;
    const int K = ptrC->col_end() - ptrC->col_start() + 1;
    
    viennacl::matrix<T> vcl_A = ptrA->device_data();
    viennacl::matrix<T> vcl_B = ptrB->device_data();
    
#ifndef BACKEND_CUDA
    viennacl::context ctx(viennacl::ocl::get_context(ptrA->getContext()));
    viennacl::matrix<T> vcl_C(M, K, ctx = ctx);
#else
    viennacl::matrix<T> vcl_C(M, K);
#endif
    
    vcl_C = viennacl::linalg::prod(vcl_A, vcl_B);
    
    ptrC->to_host(vcl_C);
}

template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else
    void
#endif 
cpp_gpuMatrix_crossprod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_)
{
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    XPtr<dynEigenMat<T> > ptrC(ptrC_);
    
    const int M = ptrC->row_end() - ptrC->row_start() + 1;
    const int K = ptrC->col_end() - ptrC->col_start() + 1;
    
    viennacl::matrix<T> vcl_A = ptrA->device_data();
    viennacl::matrix<T> vcl_B = ptrB->device_data();
    
#ifndef BACKEND_CUDA
    viennacl::context ctx(viennacl::ocl::get_context(ptrA->getContext()));
    viennacl::matrix<T> vcl_C(M, K, ctx = ctx);
#else
    int device_idx = ptrA->deviceIndex();
    cudaSetDevice(device_idx);
    
    viennacl::matrix<T> vcl_C(M, K);
#endif
    
    vcl_C = viennacl::linalg::prod(trans(vcl_A), vcl_B);
    
    ptrC->to_host(vcl_C);
}

template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else
    void
#endif 
cpp_gpuMatrix_tcrossprod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_)
{
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    XPtr<dynEigenMat<T> > ptrC(ptrC_);
    
    const int M = ptrC->row_end() - ptrC->row_start() + 1;
    const int K = ptrC->col_end() - ptrC->col_start() + 1;
    
    viennacl::matrix<T> vcl_A = ptrA->device_data();
    viennacl::matrix<T> vcl_B = ptrB->device_data();
    
#ifndef BACKEND_CUDA
    viennacl::context ctx(viennacl::ocl::get_context(ptrA->getContext()));
    viennacl::matrix<T> vcl_C(M, K, ctx = ctx);
#else
    int device_idx = ptrA->deviceIndex();
    cudaSetDevice(device_idx);
    
    viennacl::matrix<T> vcl_C(M, K);
#endif
    
    vcl_C = viennacl::linalg::prod(vcl_A, trans(vcl_B));
    
    ptrC->to_host(vcl_C);
}

template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else
    void
#endif  
cpp_gpuMatrix_transpose(
    SEXP ptrA_, 
    SEXP ptrB_)
{
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    
    const int M = ptrB->nrow();
    const int K = ptrB->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data();
    
#ifndef BACKEND_CUDA
    viennacl::context ctx(viennacl::ocl::get_context(ptrA->getContext()));
    viennacl::matrix<T> vcl_B(M, K, ctx = ctx);
#else
    int device_idx = ptrA->deviceIndex();
    cudaSetDevice(device_idx);
    
    viennacl::matrix<T> vcl_B(M, K);
#endif
    
    vcl_B = trans(vcl_A);
    
    ptrB->to_host(vcl_B);
}


/*** gpuMatrix Functions ***/

// [[Rcpp::export]]
void
cpp_gpuMatrix_gemm(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int type_flag)
{
    
    switch(type_flag) {
#ifndef BACKEND_CUDA
        case 4:
            cpp_gpuMatrix_gemm<int>(ptrA, ptrB, ptrC);
            return;
#endif
        case 6:
            cpp_gpuMatrix_gemm<float>(ptrA, ptrB, ptrC);
            return;
        case 8:
            cpp_gpuMatrix_gemm<double>(ptrA, ptrB, ptrC);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}


// [[Rcpp::export]]
void
cpp_gpuMatrix_crossprod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int type_flag)
{
    
    switch(type_flag) {
#ifndef BACKEND_CUDA
        case 4:
            cpp_gpuMatrix_crossprod<int>(ptrA, ptrB, ptrC);
            return;
#endif
        case 6:
            cpp_gpuMatrix_crossprod<float>(ptrA, ptrB, ptrC);
            return;
        case 8:
            cpp_gpuMatrix_crossprod<double>(ptrA, ptrB, ptrC);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}


// [[Rcpp::export]]
void
cpp_gpuMatrix_tcrossprod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int type_flag)
{
    
    switch(type_flag) {
#ifndef BACKEND_CUDA
        case 4:
            cpp_gpuMatrix_tcrossprod<int>(ptrA, ptrB, ptrC);
            return;
#endif
        case 6:
            cpp_gpuMatrix_tcrossprod<float>(ptrA, ptrB, ptrC);
            return;
        case 8:
            cpp_gpuMatrix_tcrossprod<double>(ptrA, ptrB, ptrC);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_transpose(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag)
{
    
    switch(type_flag) {
#ifndef BACKEND_CUDA
        case 4:
            cpp_gpuMatrix_transpose<int>(ptrA, ptrB);
            return;
#endif
        case 6:
            cpp_gpuMatrix_transpose<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_gpuMatrix_transpose<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}


/*** vclMatrix Templates ***/

template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else
    void
#endif  
    cpp_vclMatrix_gemm(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_)
{
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    Rcpp::XPtr<dynVCLMat<T> > ptrC(ptrC_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B = ptrB->data();
    viennacl::matrix_range<viennacl::matrix<T> > C = ptrC->data();

    C = viennacl::linalg::prod(A, B);
}

template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else
    void
#endif   
cpp_vclMatrix_crossprod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_)
{
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    Rcpp::XPtr<dynVCLMat<T> > ptrC(ptrC_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B = ptrB->data();
    viennacl::matrix_range<viennacl::matrix<T> > C = ptrC->data();
    
    C = viennacl::linalg::prod(trans(A), B);
}

// for use in cases where two matrices crossprod result in 1 row/column
template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else
    void
#endif   
cpp_vclMat_vclVec_crossprod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_)
{
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    Rcpp::XPtr<dynVCLVec<T> > ptrC(ptrC_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B = ptrB->data();
    viennacl::vector_range<viennacl::vector_base<T> > V = ptrC->data();
    
    viennacl::vector_base<T> tmp = V;
    
    viennacl::matrix_base<T> C(tmp.handle(),
                                A.size2(), 0, 1, A.size2(),   // row layout
                                B.size2(), 0, 1, B.size2(),   // column layout
                                true); // row-major
    
    // viennacl::matrix_base<T> pC(V.handle(),
    //                             ptrC->getPtr()->size()/B.size2(), 0, 1, ptrC->getPtr()->size()/B.size2(),   //row layout
    //                             B.size2(), 0, 1, B.size2(),   //column layout
    //                             true); // row-major
    
    // viennacl::matrix_base<T> pC(V.handle(),
    //                             30, 0, 1, 30,   //row layout
    //                             1, 0, 1, 1,   //column layout
    //                             true); // row-major
    
    // std::cout << "row_start: " << row_start << std::endl; 
    // std::cout << "row_end: " << row_end << std::endl; 
    // 
    // viennacl::range r(row_start-1, row_end);
    // viennacl::range c(0, pC.size2());
    // 
    // viennacl::matrix_range<viennacl::matrix_base<T> > C(pC, r, c);
    
    // std::cout << C << std::endl;
    
    C = viennacl::linalg::prod(trans(A), B);
    
    // std::cout << C << std::endl;
    
    // V = tmp;
}


template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else
    void
#endif  
cpp_vclMatrix_tcrossprod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_)
{    
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    Rcpp::XPtr<dynVCLMat<T> > ptrC(ptrC_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B = ptrB->data();
    viennacl::matrix_range<viennacl::matrix<T> > C = ptrC->data();
    
    C = viennacl::linalg::prod(A, trans(B));
}

template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else
    void
#endif  
cpp_vclMatrix_transpose(
    SEXP ptrA_, 
    SEXP ptrB_)
{    
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A = ptrA->data();
    
    // B = trans(A);
    ptrB->updateMatrix(trans(A));
}


/*** vclMatrix Functions ***/

// [[Rcpp::export]]
void
cpp_vclMatrix_gemm(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    int type_flag)
{
    
    switch(type_flag) {
#ifndef BACKEND_CUDA
        case 4:
            cpp_vclMatrix_gemm<int>(ptrA, ptrB, ptrC);
            return;
#endif
        case 6:
            cpp_vclMatrix_gemm<float>(ptrA, ptrB, ptrC);
            return;
        case 8:
            cpp_vclMatrix_gemm<double>(ptrA, ptrB, ptrC);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclMatrix_crossprod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    int type_flag)
{
    
    switch(type_flag) {
#ifndef BACKEND_CUDA
        case 4:
            cpp_vclMatrix_crossprod<int>(ptrA, ptrB, ptrC);
            return;
#endif
        case 6:
            cpp_vclMatrix_crossprod<float>(ptrA, ptrB, ptrC);
            return;
        case 8:
            cpp_vclMatrix_crossprod<double>(ptrA, ptrB, ptrC);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclMat_vclVec_crossprod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int type_flag)
{
    
    switch(type_flag) {
#ifndef BACKEND_CUDA
    case 4:
        cpp_vclMat_vclVec_crossprod<int>(ptrA, ptrB, ptrC);
        return;
#endif
    case 6:
        cpp_vclMat_vclVec_crossprod<float>(ptrA, ptrB, ptrC);
        return;
    case 8:
        cpp_vclMat_vclVec_crossprod<double>(ptrA, ptrB, ptrC);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclMatrix_tcrossprod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    int type_flag)
{
    
    switch(type_flag) {
#ifndef BACKEND_CUDA
        case 4:
            cpp_vclMatrix_tcrossprod<int>(ptrA, ptrB, ptrC);
            return;
#endif
        case 6:
            cpp_vclMatrix_tcrossprod<float>(ptrA, ptrB, ptrC);
            return;
        case 8:
            cpp_vclMatrix_tcrossprod<double>(ptrA, ptrB, ptrC);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclMatrix_transpose(
    SEXP ptrA, SEXP ptrB, 
    int type_flag)
{
    
    switch(type_flag) {
#ifndef BACKEND_CUDA
        case 4:
            cpp_vclMatrix_transpose<int>(ptrA, ptrB);
            return;
#endif
        case 6:
            cpp_vclMatrix_transpose<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclMatrix_transpose<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


