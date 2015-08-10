
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


template <typename T>
void cpp_vienna_crossprod(
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
    viennacl::matrix<T> vcl_C(Cm.rows(),Cm.cols());
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_C = viennacl::linalg::prod(trans(vcl_A), vcl_B);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_tcrossprod(
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
    viennacl::matrix<T> vcl_C(K,N);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_C = viennacl::linalg::prod(vcl_A, trans(vcl_B));
    
    viennacl::copy(vcl_C, Cm);
}


template void cpp_vienna_crossprod<double>(MapMat<double> &Am, MapMat<double> &Bm, MapMat<double> &Cm);
template void cpp_vienna_crossprod<float>(MapMat<float> &Am, MapMat<float> &Bm, MapMat<float> &Cm);
template void cpp_vienna_tcrossprod<double>(MapMat<double> &Am, MapMat<double> &Bm, MapMat<double> &Cm);
template void cpp_vienna_tcrossprod<float>(MapMat<float> &Am, MapMat<float> &Bm, MapMat<float> &Cm);


/*** gpuMatrix Functions ***/

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

//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_dcrossprod(
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

    cpp_vienna_crossprod(Am, Bm, Cm);
}

//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_scrossprod(
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
    
    cpp_vienna_crossprod(Am, Bm, Cm);
}

//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_dtcrossprod(
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

    cpp_vienna_tcrossprod(Am, Bm, Cm);
}

//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_stcrossprod(
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
    
    cpp_vienna_tcrossprod(Am, Bm, Cm);
}


/*** vclMatrix Functions ***/

//[[Rcpp::export]]
void cpp_vienna_vclMatrix_dgemm(SEXP ptrA_, 
                                SEXP ptrB_,
                                SEXP ptrC_)
{
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::prod(*ptrA, *ptrB);
}

//[[Rcpp::export]]
void cpp_vienna_vclMatrix_sgemm(SEXP ptrA_, 
                                SEXP ptrB_, 
                                SEXP ptrC_)
{    
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);
    
    *ptrC = viennacl::linalg::prod(*ptrA, *ptrB);
}

////[[Rcpp::export]]
//void cpp_vienna_vclMatrix_igemm(SEXP ptrA_, 
//                                SEXP ptrB_, 
//                                SEXP ptrC_)
//{    
//    Rcpp::XPtr<viennacl::matrix<int> > ptrA(ptrA_);
//    Rcpp::XPtr<viennacl::matrix<int> > ptrB(ptrB_);
//    Rcpp::XPtr<viennacl::matrix<int> > ptrC(ptrC_);
//    
////    MapMat<int> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
////    MapMat<int> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
////    MapMat<int> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
//    
//    *ptrC = viennacl::linalg::prod(*ptrA, *ptrB);
//}


//[[Rcpp::export]]
void cpp_vclMatrix_dcrossprod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_)
{
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);
    
    *ptrC = viennacl::linalg::prod(trans(*ptrA), *ptrB);
}

//[[Rcpp::export]]
void cpp_vclMatrix_scrossprod(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_)
{    
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);
    
    *ptrC = viennacl::linalg::prod(trans(*ptrA), *ptrB);
}

//[[Rcpp::export]]
void cpp_vclMatrix_dtcrossprod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_)
{
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);
    
    *ptrC = viennacl::linalg::prod(*ptrA, trans(*ptrB));
}

//[[Rcpp::export]]
void cpp_vclMatrix_stcrossprod(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_)
{   
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);
    
    *ptrC = viennacl::linalg::prod(*ptrA, trans(*ptrB));
}


