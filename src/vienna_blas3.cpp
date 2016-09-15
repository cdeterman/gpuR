

#include "gpuR/windows_check.hpp"

// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/dynEigenMat.hpp"
#include "gpuR/dynVCLMat.hpp"

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

// ViennaCL headers
#include "viennacl/ocl/backend.hpp"
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
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    XPtr<dynEigenMat<T> > ptrC(ptrC_);
    
    const int M = ptrC->row_end() - ptrC->row_start() + 1;
    const int K = ptrC->col_end() - ptrC->col_start() + 1;
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B = ptrB->device_data(ctx_id);
    viennacl::matrix<T> vcl_C(M, K, ctx = ctx);
    
    vcl_C = viennacl::linalg::prod(vcl_A, vcl_B);
    
    ptrC->to_host(vcl_C);
}

template <typename T>
void 
cpp_gpuMatrix_crossprod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_,
    int ctx_id)
{
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    XPtr<dynEigenMat<T> > ptrC(ptrC_);
    
    const int M = ptrC->row_end() - ptrC->row_start() + 1;
    const int K = ptrC->col_end() - ptrC->col_start() + 1;
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B = ptrB->device_data(ctx_id);
    viennacl::matrix<T> vcl_C(M, K, ctx = ctx);
    
    vcl_C = viennacl::linalg::prod(trans(vcl_A), vcl_B);
    
    ptrC->to_host(vcl_C);
}

template <typename T>
void 
cpp_gpuMatrix_tcrossprod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_,
    int ctx_id)
{
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    XPtr<dynEigenMat<T> > ptrC(ptrC_);
    
    const int M = ptrC->row_end() - ptrC->row_start() + 1;
    const int K = ptrC->col_end() - ptrC->col_start() + 1;
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B = ptrB->device_data(ctx_id);
    viennacl::matrix<T> vcl_C(M, K, ctx = ctx);
    
    vcl_C = viennacl::linalg::prod(vcl_A, trans(vcl_B));
    
    ptrC->to_host(vcl_C);
}

template <typename T>
void 
cpp_gpuMatrix_transpose(
    SEXP ptrA_, 
    SEXP ptrB_,
    int ctx_id)
{
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    
    const int M = ptrB->nrow();
    const int K = ptrB->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B(M, K, ctx = ctx);
    
    vcl_B = trans(vcl_A);
    
    ptrB->to_host(vcl_B);
}


/*** gpuMatrix Functions ***/

// [[Rcpp::export]]
void
cpp_gpuMatrix_gemm(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_gemm<int>(ptrA, ptrB, ptrC, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_gemm<float>(ptrA, ptrB, ptrC, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_gemm<double>(ptrA, ptrB, ptrC, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}


// [[Rcpp::export]]
void
cpp_gpuMatrix_crossprod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_crossprod<int>(ptrA, ptrB, ptrC, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_crossprod<float>(ptrA, ptrB, ptrC, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_crossprod<double>(ptrA, ptrB, ptrC, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}


// [[Rcpp::export]]
void
cpp_gpuMatrix_tcrossprod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_tcrossprod<int>(ptrA, ptrB, ptrC, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_tcrossprod<float>(ptrA, ptrB, ptrC, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_tcrossprod<double>(ptrA, ptrB, ptrC, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_transpose(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_transpose<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_transpose<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_transpose<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}


/*** vclMatrix Templates ***/

template <typename T>
void cpp_vclMatrix_gemm(
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

template <typename T>
void 
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

template <typename T>
void
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

template <typename T>
void
cpp_vclMatrix_transpose(
    SEXP ptrA_, 
    SEXP ptrB_)
{    
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B = ptrB->data();
    
    B = trans(A);
}


/*** vclMatrix Functions ***/

// [[Rcpp::export]]
void
cpp_vclMatrix_gemm(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclMatrix_gemm<int>(ptrA, ptrB, ptrC);
            return;
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
        case 4:
            cpp_vclMatrix_crossprod<int>(ptrA, ptrB, ptrC);
            return;
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
cpp_vclMatrix_tcrossprod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclMatrix_tcrossprod<int>(ptrA, ptrB, ptrC);
            return;
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
        case 4:
            cpp_vclMatrix_transpose<int>(ptrA, ptrB);
            return;
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


