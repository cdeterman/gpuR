
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

using namespace Rcpp;

template <typename T>
void cpp_vienna_gemm(
    MapMat<T> &Am, 
    MapMat<T> &Bm, 
    MapMat<T> &Cm)
{    
    //use only GPUs:
    long id = 0;
    viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    
    int M = Am.cols();
    int K = Am.rows();
    int N = Bm.rows();
    int P = Bm.cols();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(N,P);
    viennacl::matrix<T> vcl_C(K,P);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_C = viennacl::linalg::prod(vcl_A, vcl_B);
    
    viennacl::copy(vcl_C, Cm);
}


//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_dgemm(SEXP ptrA_, 
                                SEXP ptrB_,
                                SEXP ptrC_)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_gemm<double>(Am, Bm, Cm);
}

//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_sgemm(SEXP ptrA_, 
                                SEXP ptrB_, 
                                SEXP ptrC_)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_gemm<float>(Am, Bm, Cm);
}

//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_igemm(SEXP ptrA_, 
                                SEXP ptrB_, 
                                SEXP ptrC_)
{    
    Rcpp::XPtr<dynEigen<int> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<int> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<int> > ptrC(ptrC_);
    
    MapMat<int> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<int> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<int> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_gemm<int>(Am, Bm, Cm);
}

