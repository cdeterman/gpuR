
// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/eigen_templates.hpp"
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
#include "viennacl/linalg/sum.hpp"

using namespace Rcpp;

template <typename T>
void 
cpp_gpuMatrix_pmcc(
    SEXP ptrA_, 
    SEXP ptrB_,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<dynEigen<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<T> > ptrB(ptrB_);
    
    MapMat<T> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<T> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    
    const int M = Am.cols();
    const int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::vector<T> ones = viennacl::scalar_vector<T>(K, 1);
    viennacl::vector<T> vcl_meanVec(M);
    viennacl::matrix<T> vcl_meanMat(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    // vector of column means
    vcl_meanVec = viennacl::linalg::column_sum(vcl_A);
    vcl_meanVec *= (T)(1)/(T)(K);
    
    // matrix of means
    vcl_meanMat = viennacl::linalg::outer_prod(ones, vcl_meanVec);
    
    viennacl::matrix<T> tmp = vcl_A - vcl_meanMat;
    
    // calculate pearson covariance
    viennacl::matrix<T> vcl_B = viennacl::linalg::prod(trans(tmp), tmp);
    vcl_B *= (T)(1)/(T)(K-1);
    
    viennacl::copy(vcl_B, Bm);
}

template <typename T>
void 
cpp_vclMatrix_pmcc(
    SEXP ptrA_, 
    SEXP ptrB_,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<T> > ptrB(ptrB_);
    
    viennacl::matrix<T> &vcl_A = *ptrA;
    viennacl::matrix<T> &vcl_B = *ptrB;
    
    const int M = vcl_A.size2();
    const int K = vcl_A.size1();
    
    viennacl::vector<T> ones = viennacl::scalar_vector<T>(K, 1);
    viennacl::vector<T> vcl_meanVec(M);
    viennacl::matrix<T> vcl_meanMat(K,M);
    
    // vector of column means
    vcl_meanVec = viennacl::linalg::column_sum(vcl_A);
    vcl_meanVec *= (T)(1)/(T)(K);
    
    // matrix of means
    vcl_meanMat = viennacl::linalg::outer_prod(ones, vcl_meanVec);
    
    viennacl::matrix<T> tmp = vcl_A - vcl_meanMat;
    
    // calculate pearson covariance
    vcl_B = viennacl::linalg::prod(trans(tmp), tmp);
    vcl_B *= (T)(1)/(T)(K-1);
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_pmcc(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_pmcc<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_pmcc<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_pmcc<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclMatrix_pmcc(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclMatrix_pmcc<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclMatrix_pmcc<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclMatrix_pmcc<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

