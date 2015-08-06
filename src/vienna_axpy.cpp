
// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/eigen_templates.hpp"
#include "gpuR/dynEigen.hpp"

using namespace Rcpp;

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

// ViennaCL headers
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/matrix.hpp"

using namespace Rcpp;

template <typename T>
void cpp_arma_vienna_axpy(
    T const alpha, 
    MapMat<T> &Am, 
    MapMat<T> &Bm)
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
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_B += alpha * vcl_A;

    viennacl::copy(vcl_B, Bm);
}

template void cpp_arma_vienna_axpy<double>(double const alpha, MapMat<double> &Am, MapMat<double> &Bm);
template void cpp_arma_vienna_axpy<float>(float const alpha, MapMat<float> &Am, MapMat<float> &Bm);

//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_daxpy(SEXP alpha_, 
                                SEXP ptrA_, 
                                SEXP ptrB_)
{
    const double alpha = as<double>(alpha_);
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrB(ptrB_);
    
    MapMat<double> Am = MapMat<double>(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Bm = MapMat<double>(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    
    cpp_arma_vienna_axpy(alpha, Am, Bm);
}

//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_saxpy(SEXP alpha_, 
                                SEXP ptrA_, 
                                SEXP ptrB_)
{
    const float alpha = as<float>(alpha_);

    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrB(ptrB_);
    
    MapMat<float> Am = MapMat<float>(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Bm = MapMat<float>(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    
    cpp_arma_vienna_axpy(alpha, Am, Bm);
}
