// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/eigen_templates.hpp"
#include "gpuR/dynEigen.hpp"
#include "gpuR/dynEigenVec.hpp"

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

// ViennaCL headers
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/sum.hpp"

using namespace Rcpp;

/*** gpuMatrix Templates ***/

template <typename T>
void 
cpp_gpuMatrix_colmean(
    SEXP ptrA_, 
    SEXP ptrC_,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<dynEigen<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    MapMat<T> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapVec<T> colMeans(ptrC->ptr(), ptrC->length());
    
    const int M = Am.cols();
    const int K = Am.rows();
    const int V = colMeans.size();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::vector<T> vcl_colMeans(V);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_colMeans = viennacl::linalg::column_sum(vcl_A);
    vcl_colMeans *= (T)(1)/(T)(K);
    
    viennacl::copy(vcl_colMeans, colMeans);
}

template <typename T>
void 
cpp_gpuMatrix_colsum(
    SEXP ptrA_, SEXP ptrC_,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<dynEigen<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    MapMat<T> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapVec<T> colSums(ptrC->ptr(), ptrC->length());
    
    const int M = Am.cols();
    const int K = Am.rows();
    const int V = colSums.size();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::vector<T> vcl_colSums(V);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_colSums = viennacl::linalg::column_sum(vcl_A);
    
    viennacl::copy(vcl_colSums, colSums);
}

template <typename T>
void 
cpp_gpuMatrix_rowmean(
    SEXP ptrA_, SEXP ptrC_,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<dynEigen<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    MapMat<T> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapVec<T> rowMeans(ptrC->ptr(), ptrC->length());

    const int M = Am.cols();
    const int K = Am.rows();
    const int V = rowMeans.size();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::vector<T> vcl_rowMeans(V);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_rowMeans = viennacl::linalg::row_sum(vcl_A);
    vcl_rowMeans *= (T)(1)/(T)(M);
    
    viennacl::copy(vcl_rowMeans, rowMeans);
}

template <typename T>
void
cpp_gpuMatrix_rowsum(
    SEXP ptrA_, SEXP ptrC_,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<dynEigen<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    MapMat<T> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapVec<T> rowSums(ptrC->ptr(), ptrC->length());

    const int M = Am.cols();
    const int K = Am.rows();
    const int V = rowSums.size();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::vector<T> vcl_rowSums(V);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_rowSums = viennacl::linalg::row_sum(vcl_A);
    
    viennacl::copy(vcl_rowSums, rowSums);
}

/*** vclMatrix Templates ***/

template <typename T>
void 
cpp_vclMatrix_colmean(
    SEXP ptrA_, 
    SEXP ptrC_,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrC(ptrC_);
    
    viennacl::matrix<T> &vcl_A = *ptrA;
    viennacl::vector<T> &vcl_colMeans = *ptrC;
    
    const int K = vcl_A.size1();
        
    vcl_colMeans = viennacl::linalg::column_sum(vcl_A);
    vcl_colMeans *= (T)(1)/(T)(K);
}

template <typename T>
void 
cpp_vclMatrix_colsum(
    SEXP ptrA_, SEXP ptrC_,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrC(ptrC_);
    
    viennacl::matrix<T> &vcl_A = *ptrA;
    viennacl::vector<T> &vcl_colSums = *ptrC;
    
    vcl_colSums = viennacl::linalg::column_sum(vcl_A);
}

template <typename T>
void 
cpp_vclMatrix_rowmean(
    SEXP ptrA_, SEXP ptrC_,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrC(ptrC_);
    
    viennacl::matrix<T> &vcl_A = *ptrA;
    viennacl::vector<T> &vcl_rowMeans = *ptrC;

    const int M = vcl_A.size2();
    
    vcl_rowMeans = viennacl::linalg::row_sum(vcl_A);
    vcl_rowMeans *= (T)(1)/(T)(M);
}

template <typename T>
void
cpp_vclMatrix_rowsum(
    SEXP ptrA_, SEXP ptrC_,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrC(ptrC_);
    
    viennacl::matrix<T> &vcl_A = *ptrA;
    viennacl::vector<T> &vcl_rowSums = *ptrC;
    
    vcl_rowSums = viennacl::linalg::row_sum(vcl_A);
}

/*** gpuMatrix Functions ***/

// [[Rcpp::export]]
void
cpp_gpuMatrix_colmean(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_colmean<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_colmean<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_colmean<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_colsum(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_colsum<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_colsum<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_colsum<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_rowmean(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_rowmean<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_rowmean<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_rowmean<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_gpuMatrix_rowsum(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_rowsum<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_rowsum<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_rowsum<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

/*** vclMatrix Functions ***/

// [[Rcpp::export]]
void
cpp_vclMatrix_colmean(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclMatrix_colmean<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclMatrix_colmean<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclMatrix_colmean<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclMatrix_colsum(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclMatrix_colsum<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclMatrix_colsum<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclMatrix_colsum<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclMatrix_rowmean(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclMatrix_rowmean<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclMatrix_rowmean<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclMatrix_rowmean<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclMatrix_rowsum(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclMatrix_rowsum<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclMatrix_rowsum<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclMatrix_rowsum<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}
