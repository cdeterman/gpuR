
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

/*** templates ***/

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

template <typename T>
void cpp_vienna_elem_prod(
    MapMat<T> &Am, 
    MapMat<T> &Bm, 
    MapMat<T> &Cm)
{    
    //use only GPUs:
    long id = 0;
    viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(K,M);
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_C = viennacl::linalg::element_prod(vcl_A, vcl_B);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_elem_div(
    MapMat<T> &Am, 
    MapMat<T> &Bm, 
    MapMat<T> &Cm)
{    
    //use only GPUs:
    long id = 0;
    viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(K,M);
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_C = viennacl::linalg::element_div(vcl_A, vcl_B);
    
    viennacl::copy(vcl_C, Cm);
}


template <typename T>
void cpp_vienna_elem_sin(
    MapMat<T> &Am, 
    MapMat<T> &Cm)
{    
    //use only GPUs:
    long id = 0;
    viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_sin(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_elem_cos(
    MapMat<T> &Am, 
    MapMat<T> &Cm)
{    
    //use only GPUs:
    long id = 0;
    viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_cos(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_elem_tan(
    MapMat<T> &Am, 
    MapMat<T> &Cm)
{    
    //use only GPUs:
    long id = 0;
    viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_tan(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

/*** gpuMatrix functions ***/

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

//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_elem_prod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_prod<double>(Am, Bm, Cm);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuMatrix_elem_prod(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_elem_prod<float>(Am, Bm, Cm);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_elem_div(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_div<double>(Am, Bm, Cm);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuMatrix_elem_div(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_elem_div<float>(Am, Bm, Cm);
}

/*** vclMatrix Functions ***/

//[[Rcpp::export]]
void cpp_vclMatrix_daxpy(SEXP alpha_, 
                        SEXP ptrA_, 
                        SEXP ptrB_)
{
    const double alpha = as<double>(alpha_);
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrB(ptrB_);
    
    *ptrB += alpha * (*ptrA);
}

//[[Rcpp::export]]
void cpp_vclMatrix_saxpy(SEXP alpha_, 
                            SEXP ptrA_, 
                            SEXP ptrB_)
{
    const float alpha = as<float>(alpha_);

    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrB(ptrB_);
    
    *ptrB += alpha * (*ptrA);
}


//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_elem_sin(
    SEXP ptrA_, 
    SEXP ptrC_)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_sin<double>(Am, Cm);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuMatrix_elem_sin(
    SEXP ptrA_, 
    SEXP ptrC_)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_elem_sin<float>(Am, Cm);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_elem_cos(
    SEXP ptrA_, 
    SEXP ptrC_)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_cos<double>(Am, Cm);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuMatrix_elem_cos(
    SEXP ptrA_, 
    SEXP ptrC_)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_elem_cos<float>(Am, Cm);
}


//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_elem_tan(
    SEXP ptrA_, 
    SEXP ptrC_)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_tan<double>(Am, Cm);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuMatrix_elem_tan(
    SEXP ptrA_, 
    SEXP ptrC_)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_elem_tan<float>(Am, Cm);
}

