
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
#include "viennacl/vector.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/inner_prod.hpp"

using namespace Rcpp;

/*** templates ***/


/*** gpuVector Templates ***/

template <typename T>
void cpp_gpuVector_axpy(
    SEXP alpha_, 
    SEXP A_, SEXP B_,
    int device_flag)
{
    if(device_flag == 0){
        //use only GPUs:
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    const T alpha = as<T>(alpha_);
    Rcpp::XPtr<dynEigenVec<T> > ptrA(A_);
    Rcpp::XPtr<dynEigenVec<T> > ptrB(B_);
    
    MapVec<T> Am(ptrA->ptr(), ptrA->length());
    MapVec<T> Bm(ptrB->ptr(), ptrB->length());
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_B(M);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_B += alpha * vcl_A;
    
    viennacl::copy(vcl_B, Bm);
}

template <typename T>
T cpp_gpuVector_inner_prod(
    SEXP ptrA_, 
    SEXP ptrB_, 
    const int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    T C;
    
    Rcpp::XPtr<dynEigenVec<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<T> > ptrB(ptrB_);
    
    MapVec<T> Am(ptrA->ptr(), ptrA->length());
    MapVec<T> Bm(ptrB->ptr(), ptrB->length());
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_B(M);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    C = viennacl::linalg::inner_prod(vcl_A, vcl_B);
    
    return C;
}

template <typename T>
void cpp_gpuVector_outer_prod(
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
    
    Rcpp::XPtr<dynEigenVec<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<T> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<T> > ptrC(ptrC_);
    
    MapVec<T> Am(ptrA->ptr(), ptrA->length());
    MapVec<T> Bm(ptrB->ptr(), ptrB->length());
    MapMat<T> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_B(M);
    viennacl::matrix<T> vcl_C(M, M);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_C = viennacl::linalg::outer_prod(vcl_A, vcl_B);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_gpuVector_elem_prod(
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
    
    Rcpp::XPtr<dynEigenVec<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<T> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    MapVec<T> Am(ptrA->ptr(), ptrA->length());
    MapVec<T> Bm(ptrB->ptr(), ptrB->length());
    MapVec<T> Cm(ptrC->ptr(), ptrC->length());
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_B(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_C = viennacl::linalg::element_prod(vcl_A, vcl_B);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_gpuVector_elem_div(
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
    
    Rcpp::XPtr<dynEigenVec<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<T> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    MapVec<T> Am(ptrA->ptr(), ptrA->length());
    MapVec<T> Bm(ptrB->ptr(), ptrB->length());
    MapVec<T> Cm(ptrC->ptr(), ptrC->length());
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_B(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_C = viennacl::linalg::element_div(vcl_A, vcl_B);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_sin(
    SEXP ptrA_, SEXP ptrC_,
    const int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<dynEigenVec<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    MapVec<T> Am(ptrA->ptr(), ptrA->length());
    MapVec<T> Cm(ptrC->ptr(), ptrC->length());
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_sin(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_asin(
    SEXP ptrA_, SEXP ptrC_,
    const int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<dynEigenVec<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    MapVec<T> Am(ptrA->ptr(), ptrA->length());
    MapVec<T> Cm(ptrC->ptr(), ptrC->length());
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_asin(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_sinh(
    SEXP ptrA_, SEXP ptrC_,
    const int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<dynEigenVec<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    MapVec<T> Am(ptrA->ptr(), ptrA->length());
    MapVec<T> Cm(ptrC->ptr(), ptrC->length());
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_sinh(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_cos(
    SEXP ptrA_, SEXP ptrC_,
    const int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<dynEigenVec<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    MapVec<T> Am(ptrA->ptr(), ptrA->length());
    MapVec<T> Cm(ptrC->ptr(), ptrC->length());
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_cos(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_acos(
    SEXP ptrA_, SEXP ptrC_,
    const int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<dynEigenVec<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    MapVec<T> Am(ptrA->ptr(), ptrA->length());
    MapVec<T> Cm(ptrC->ptr(), ptrC->length());
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_acos(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_cosh(
    SEXP ptrA_, SEXP ptrC_,
    const int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<dynEigenVec<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    MapVec<T> Am(ptrA->ptr(), ptrA->length());
    MapVec<T> Cm(ptrC->ptr(), ptrC->length());
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_cosh(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_tan(
    SEXP ptrA_, SEXP ptrC_,
    const int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<dynEigenVec<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    MapVec<T> Am(ptrA->ptr(), ptrA->length());
    MapVec<T> Cm(ptrC->ptr(), ptrC->length());
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_tan(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_atan(
    SEXP ptrA_, SEXP ptrC_,
    const int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<dynEigenVec<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    MapVec<T> Am(ptrA->ptr(), ptrA->length());
    MapVec<T> Cm(ptrC->ptr(), ptrC->length());
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_atan(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_tanh(
    SEXP ptrA_, SEXP ptrC_,
    const int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<dynEigenVec<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    MapVec<T> Am(ptrA->ptr(), ptrA->length());
    MapVec<T> Cm(ptrC->ptr(), ptrC->length());
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_tanh(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

/*** gpuMatrix Templates ***/

template <typename T>
void 
cpp_gpuMatrix_axpy(
    SEXP alpha_, 
    SEXP ptrA_, 
    SEXP ptrB_,
    const int device_flag)
{
         //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    const T alpha = as<T>(alpha_);
    
    Rcpp::XPtr<dynEigen<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<T> > ptrB(ptrB_);
    
    MapMat<T> Am = MapMat<T>(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<T> Bm = MapMat<T>(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    
    const int M = Am.cols();
    const int K = Am.rows();
    const int N = Bm.rows();
    const int P = Bm.cols();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(N,P);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_B += alpha * vcl_A;

    viennacl::copy(vcl_B, Bm);
}

template <typename T>
void cpp_gpuMatrix_elem_prod(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_, 
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<dynEigen<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<T> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<T> > ptrC(ptrC_);
    
    MapMat<T> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<T> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<T> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
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
void cpp_gpuMatrix_elem_div(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_, 
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<dynEigen<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<T> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<T> > ptrC(ptrC_);
    
    MapMat<T> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<T> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<T> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
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
void cpp_gpuMatrix_elem_sin(
    SEXP ptrA_, 
    SEXP ptrB_, 
    int device_flag)
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
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_B = viennacl::linalg::element_sin(vcl_A);
    
    viennacl::copy(vcl_B, Bm);
}

template <typename T>
void cpp_gpuMatrix_elem_asin(
    SEXP ptrA_, 
    SEXP ptrB_, 
    int device_flag)
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
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_B = viennacl::linalg::element_asin(vcl_A);
    
    viennacl::copy(vcl_B, Bm);
}


template <typename T>
void cpp_gpuMatrix_elem_sinh(
    SEXP ptrA_, 
    SEXP ptrB_, 
    int device_flag)
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
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_B = viennacl::linalg::element_sinh(vcl_A);
    
    viennacl::copy(vcl_B, Bm);
}


template <typename T>
void cpp_gpuMatrix_elem_cos(
    SEXP ptrA_, 
    SEXP ptrB_, 
    int device_flag)
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
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_B = viennacl::linalg::element_cos(vcl_A);
    
    viennacl::copy(vcl_B, Bm);
}

template <typename T>
void cpp_gpuMatrix_elem_acos(
    SEXP ptrA_, 
    SEXP ptrB_, 
    int device_flag)
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
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_B = viennacl::linalg::element_acos(vcl_A);
    
    viennacl::copy(vcl_B, Bm);
}


template <typename T>
void cpp_gpuMatrix_elem_cosh(
    SEXP ptrA_, 
    SEXP ptrB_, 
    int device_flag)
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
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_B = viennacl::linalg::element_cosh(vcl_A);
    
    viennacl::copy(vcl_B, Bm);
}


template <typename T>
void cpp_gpuMatrix_elem_tan(
    SEXP ptrA_, 
    SEXP ptrB_, 
    int device_flag)
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
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_B = viennacl::linalg::element_tan(vcl_A);
    
    viennacl::copy(vcl_B, Bm);
}

template <typename T>
void cpp_gpuMatrix_elem_atan(
    SEXP ptrA_, 
    SEXP ptrB_, 
    int device_flag)
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
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_B = viennacl::linalg::element_atan(vcl_A);
    
    viennacl::copy(vcl_B, Bm);
}


template <typename T>
void cpp_gpuMatrix_elem_tanh(
    SEXP ptrA_, 
    SEXP ptrB_, 
    int device_flag)
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
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_B = viennacl::linalg::element_tanh(vcl_A);
    
    viennacl::copy(vcl_B, Bm);
}

template <typename T>
void cpp_gpuMatrix_elem_log(
    SEXP ptrA_, SEXP ptrB_,
    int device_flag)
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
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_B = viennacl::linalg::element_log(vcl_A);
    
    viennacl::copy(vcl_B, Bm);
}

template <typename T>
void cpp_gpuMatrix_elem_log_base(
    SEXP ptrA_, SEXP ptrB_,
    T base,
    int device_flag)
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
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_B = viennacl::linalg::element_log10(vcl_A);
    vcl_B /= log10(base);
    
    viennacl::copy(vcl_B, Bm);
}

template <typename T>
void cpp_gpuMatrix_elem_log10(
    SEXP ptrA_, SEXP ptrB_,
    int device_flag)
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
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_B = viennacl::linalg::element_log10(vcl_A);
    
    viennacl::copy(vcl_B, Bm);
}

template <typename T>
void cpp_gpuMatrix_elem_exp(
    SEXP ptrA_, SEXP ptrB_,
    int device_flag)
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
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_B = viennacl::linalg::element_exp(vcl_A);
    
    viennacl::copy(vcl_B, Bm);
}

/*** gpuMatrix Functions ***/

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_prod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_prod<int>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_elem_prod<float>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_elem_prod<double>(ptrA, ptrB, ptrC, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_div(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_div<int>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_elem_div<float>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_elem_div<double>(ptrA, ptrB, ptrC, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_sin(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_sin<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_elem_sin<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_elem_sin<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_asin(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_asin<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_elem_asin<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_elem_asin<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_sinh(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_sinh<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_elem_sinh<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_elem_sinh<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_cos(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_cos<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_elem_cos<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_elem_cos<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_acos(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_acos<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_elem_acos<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_elem_acos<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_cosh(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_cosh<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_elem_cosh<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_elem_cosh<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_tan(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_tan<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_elem_tan<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_elem_tan<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_atan(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_atan<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_elem_atan<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_elem_atan<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_tanh(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_tanh<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_elem_tanh<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_elem_tanh<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_log(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_log<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_elem_log<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_elem_log<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_log_base(
    SEXP ptrA, SEXP ptrB,
    SEXP base,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_log_base<int>(ptrA, ptrB, as<int>(base), device_flag);
            return;
        case 6:
            cpp_gpuMatrix_elem_log_base<float>(ptrA, ptrB, as<float>(base), device_flag);
            return;
        case 8:
            cpp_gpuMatrix_elem_log_base<double>(ptrA, ptrB, as<double>(base), device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_log10(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_log10<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_elem_log10<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_elem_log10<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_exp(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_exp<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_elem_exp<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_elem_exp<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_axpy(
    SEXP alpha,
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_axpy<int>(alpha, ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_axpy<float>(alpha, ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_axpy<double>(alpha, ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


/*** vclVector Templates ***/

template <typename T>
void cpp_vclVector_axpy(
    SEXP alpha_, 
    SEXP ptrA_, 
    SEXP ptrB_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    const T alpha = as<T>(alpha_);

    Rcpp::XPtr<viennacl::vector<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrB(ptrB_);
    
    *ptrB += alpha * (*ptrA);
}


template <typename T>
T cpp_vclVector_inner_prod(
    SEXP ptrA_, 
    SEXP ptrB_,
    int device_flag)
{
    float out;
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrB(ptrB_);
    
    out = viennacl::linalg::inner_prod(*ptrA, *ptrB);
    return out;
}


template <typename T>
void cpp_vclVector_outer_prod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<T> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::outer_prod(*ptrA, *ptrB);
}


template <typename T>
void cpp_vclVector_elem_prod(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::vector<T> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_prod(*ptrA, *ptrB);
}

template <typename T>
void cpp_vclVector_elem_div(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::vector<T> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_div(*ptrA, *ptrB);
}


template <typename T>
void cpp_vclVector_elem_sin(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_sin(*ptrA);
}


template <typename T>
void cpp_vclVector_elem_asin(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_asin(*ptrA);
}


template <typename T>
void cpp_vclVector_elem_sinh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_sinh(*ptrA);
}


template <typename T>
void cpp_vclVector_elem_cos(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_cos(*ptrA);
}


template <typename T>
void cpp_vclVector_elem_acos(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_acos(*ptrA);
}


template <typename T>
void cpp_vclVector_elem_cosh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_cosh(*ptrA);
}


template <typename T>
void cpp_vclVector_elem_tan(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_tan(*ptrA);
}


template <typename T>
void cpp_vclVector_elem_atan(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_atan(*ptrA);
}


template <typename T>
void cpp_vclVector_elem_tanh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_tanh(*ptrA);
}

template <typename T>
void cpp_vclVector_elem_exp(
    SEXP ptrA_, 
    SEXP ptrC_,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_exp(*ptrA);
}


template <typename T>
void cpp_vclVector_elem_log10(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{ 
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_log10(*ptrA);
}


template <typename T>
void cpp_vclVector_elem_log_base(
    SEXP ptrA_, 
    SEXP ptrC_,
    T base,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_log10(*ptrA);
    *ptrC /= log10(base);
}

template <typename T>
void cpp_vclVector_elem_log(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<T> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_log(*ptrA);
}

/*** vclMatrix templates ***/

template <typename T>
void cpp_vclMatrix_axpy(
    SEXP alpha_, 
    SEXP ptrA_, 
    SEXP ptrB_,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    const T alpha = as<T>(alpha_);
    
    Rcpp::XPtr<viennacl::matrix<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<T> > ptrB(ptrB_);
    
    *ptrB += alpha * (*ptrA);
}

template <typename T>
void cpp_vclMatrix_elem_prod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<T> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<T> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_prod(*ptrA, *ptrB);
}

template <typename T>
void cpp_vclMatrix_elem_div(
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

    *ptrC = viennacl::linalg::element_div(*ptrA, *ptrB);
}

template <typename T>
void cpp_vclMatrix_elem_sin(
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

    *ptrB = viennacl::linalg::element_sin(*ptrA);
}

template <typename T>
void cpp_vclMatrix_elem_asin(
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

    *ptrB = viennacl::linalg::element_asin(*ptrA);
}

template <typename T>
void cpp_vclMatrix_elem_sinh(
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

    *ptrB = viennacl::linalg::element_sinh(*ptrA);
}

template <typename T>
void cpp_vclMatrix_elem_cos(
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

    *ptrB = viennacl::linalg::element_cos(*ptrA);
}

template <typename T>
void cpp_vclMatrix_elem_acos(
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

    *ptrB = viennacl::linalg::element_acos(*ptrA);
}

template <typename T>
void cpp_vclMatrix_elem_cosh(
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

    *ptrB = viennacl::linalg::element_cosh(*ptrA);
}

template <typename T>
void cpp_vclMatrix_elem_tan(
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

    *ptrB = viennacl::linalg::element_tan(*ptrA);
}

template <typename T>
void cpp_vclMatrix_elem_atan(
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

    *ptrB = viennacl::linalg::element_atan(*ptrA);
}

template <typename T>
void cpp_vclMatrix_elem_tanh(
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

    *ptrB = viennacl::linalg::element_tanh(*ptrA);
}

template <typename T>
void cpp_vclMatrix_elem_log(
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

    *ptrB = viennacl::linalg::element_log(*ptrA);
}


template <typename T>
void cpp_vclMatrix_elem_log10(
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

    *ptrB = viennacl::linalg::element_log10(*ptrA);
}


template <typename T>
void cpp_vclMatrix_elem_log_base(
    SEXP ptrA_, 
    SEXP ptrB_,
    const float base,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<T> > ptrB(ptrB_);

    *ptrB = viennacl::linalg::element_log10(*ptrA);
    *ptrB /= log10(base);
}

template <typename T>
void cpp_vclMatrix_elem_exp(
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

    *ptrB = viennacl::linalg::element_exp(*ptrA);
}

/*** vclMatrix Functions ***/

// [[Rcpp::export]]
void
cpp_vclMatrix_axpy(
    SEXP alpha,
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclMatrix_axpy<int>(alpha, ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclMatrix_axpy<float>(alpha, ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclMatrix_axpy<double>(alpha, ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


//[[Rcpp::export]]
void cpp_vclMatrix_elem_prod(
    SEXP ptrA, 
    SEXP ptrB,
    SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_prod<int>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 6:
            cpp_vclMatrix_elem_prod<float>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 8:
            cpp_vclMatrix_elem_prod<double>(ptrA, ptrB, ptrC, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

//[[Rcpp::export]]
void cpp_vclMatrix_elem_div(
    SEXP ptrA, 
    SEXP ptrB,
    SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_div<int>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 6:
            cpp_vclMatrix_elem_div<float>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 8:
            cpp_vclMatrix_elem_div<double>(ptrA, ptrB, ptrC, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

//[[Rcpp::export]]
void cpp_vclMatrix_elem_sin(
    SEXP ptrA, 
    SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_sin<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclMatrix_elem_sin<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclMatrix_elem_sin<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

//[[Rcpp::export]]
void cpp_vclMatrix_elem_asin(
    SEXP ptrA, 
    SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_asin<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclMatrix_elem_asin<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclMatrix_elem_asin<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

//[[Rcpp::export]]
void cpp_vclMatrix_elem_sinh(
    SEXP ptrA, 
    SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_sinh<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclMatrix_elem_sinh<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclMatrix_elem_sinh<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


//[[Rcpp::export]]
void cpp_vclMatrix_elem_cos(
    SEXP ptrA, 
    SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_cos<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclMatrix_elem_cos<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclMatrix_elem_cos<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

//[[Rcpp::export]]
void cpp_vclMatrix_elem_acos(
    SEXP ptrA, 
    SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_acos<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclMatrix_elem_acos<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclMatrix_elem_acos<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


//[[Rcpp::export]]
void cpp_vclMatrix_elem_cosh(
    SEXP ptrA, 
    SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_cosh<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclMatrix_elem_cosh<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclMatrix_elem_cosh<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


//[[Rcpp::export]]
void cpp_vclMatrix_elem_tan(
    SEXP ptrA, 
    SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_tan<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclMatrix_elem_tan<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclMatrix_elem_tan<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

//[[Rcpp::export]]
void cpp_vclMatrix_elem_atan(
    SEXP ptrA, 
    SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_atan<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclMatrix_elem_atan<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclMatrix_elem_atan<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

//[[Rcpp::export]]
void cpp_vclMatrix_elem_tanh(
    SEXP ptrA, 
    SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_tanh<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclMatrix_elem_tanh<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclMatrix_elem_tanh<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

//[[Rcpp::export]]
void cpp_vclMatrix_elem_log(
    SEXP ptrA, 
    SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_log<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclMatrix_elem_log<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclMatrix_elem_log<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


//[[Rcpp::export]]
void cpp_vclMatrix_elem_log10(
    SEXP ptrA, 
    SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_log10<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclMatrix_elem_log10<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclMatrix_elem_log10<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


//[[Rcpp::export]]
void cpp_vclMatrix_elem_log_base(
    SEXP ptrA, 
    SEXP ptrB,
    SEXP base,
    const int device_flag,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_log_base<int>(ptrA, ptrB, as<int>(base), device_flag);
            return;
        case 6:
            cpp_vclMatrix_elem_log_base<float>(ptrA, ptrB, as<float>(base), device_flag);
            return;
        case 8:
            cpp_vclMatrix_elem_log_base<double>(ptrA, ptrB, as<double>(base), device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

//[[Rcpp::export]]
void cpp_vclMatrix_elem_exp(
    SEXP ptrA, 
    SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_exp<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclMatrix_elem_exp<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclMatrix_elem_exp<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


/*** gpuVector functions ***/

// [[Rcpp::export]]
void
cpp_gpuVector_axpy(
    SEXP alpha,
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_axpy<int>(alpha, ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuVector_axpy<float>(alpha, ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuVector_axpy<double>(alpha, ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
SEXP
cpp_gpuVector_inner_prod(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            return wrap(cpp_gpuVector_inner_prod<int>(ptrA, ptrB, device_flag));
        case 6:
            return wrap(cpp_gpuVector_inner_prod<float>(ptrA, ptrB, device_flag));
        case 8:
            return wrap(cpp_gpuVector_inner_prod<double>(ptrA, ptrB, device_flag));
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_outer_prod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_outer_prod<int>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 6:
            cpp_gpuVector_outer_prod<float>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 8:
            cpp_gpuVector_outer_prod<double>(ptrA, ptrB, ptrC, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_gpuVector_elem_prod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_prod<int>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 6:
            cpp_gpuVector_elem_prod<float>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 8:
            cpp_gpuVector_elem_prod<double>(ptrA, ptrB, ptrC, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_div(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_div<int>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 6:
            cpp_gpuVector_elem_div<float>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 8:
            cpp_gpuVector_elem_div<double>(ptrA, ptrB, ptrC, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_sin(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_sin<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuVector_elem_sin<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuVector_elem_sin<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_asin(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_asin<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuVector_elem_asin<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuVector_elem_asin<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_sinh(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_sinh<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuVector_elem_sinh<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuVector_elem_sinh<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_cos(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_cos<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuVector_elem_cos<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuVector_elem_cos<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_acos(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_acos<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuVector_elem_acos<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuVector_elem_acos<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_cosh(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_cosh<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuVector_elem_cosh<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuVector_elem_cosh<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_gpuVector_elem_tan(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_tan<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuVector_elem_tan<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuVector_elem_tan<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_atan(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_atan<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuVector_elem_atan<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuVector_elem_atan<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_tanh(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_tanh<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuVector_elem_tanh<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuVector_elem_tanh<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}



/*** vclVector Functions ***/

// [[Rcpp::export]]
void
cpp_vclVector_axpy(
    SEXP alpha,
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_axpy<int>(alpha, ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclVector_axpy<float>(alpha, ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclVector_axpy<double>(alpha, ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
SEXP
cpp_vclVector_inner_prod(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            return wrap(cpp_vclVector_inner_prod<int>(ptrA, ptrB, device_flag));
        case 6:
            return wrap(cpp_vclVector_inner_prod<float>(ptrA, ptrB, device_flag));
        case 8:
            return wrap(cpp_vclVector_inner_prod<double>(ptrA, ptrB, device_flag));
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_outer_prod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_outer_prod<int>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 6:
            cpp_vclVector_outer_prod<float>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 8:
            cpp_vclVector_outer_prod<double>(ptrA, ptrB, ptrC, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_elem_prod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_prod<int>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 6:
            cpp_vclVector_elem_prod<float>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 8:
            cpp_vclVector_elem_prod<double>(ptrA, ptrB, ptrC, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_elem_div(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_div<int>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 6:
            cpp_vclVector_elem_div<float>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 8:
            cpp_vclVector_elem_div<double>(ptrA, ptrB, ptrC, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_elem_sin(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_sin<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclVector_elem_sin<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclVector_elem_sin<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_elem_asin(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_asin<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclVector_elem_asin<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclVector_elem_asin<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclVector_elem_sinh(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_sinh<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclVector_elem_sinh<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclVector_elem_sinh<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclVector_elem_cos(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_cos<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclVector_elem_cos<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclVector_elem_cos<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_elem_acos(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_acos<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclVector_elem_acos<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclVector_elem_acos<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclVector_elem_cosh(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_cosh<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclVector_elem_cosh<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclVector_elem_cosh<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclVector_elem_tan(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_tan<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclVector_elem_tan<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclVector_elem_tan<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_elem_atan(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_atan<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclVector_elem_atan<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclVector_elem_atan<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclVector_elem_tanh(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_tanh<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclVector_elem_tanh<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclVector_elem_tanh<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclVector_elem_log(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_log<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclVector_elem_log<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclVector_elem_log<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_elem_log10(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_log10<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclVector_elem_log10<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclVector_elem_log10<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclVector_elem_log_base(
    SEXP ptrA, SEXP ptrB, 
    SEXP R_base,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_log_base<int>(ptrA, ptrB, as<int>(R_base), device_flag);
            return;
        case 6:
            cpp_vclVector_elem_log_base<float>(ptrA, ptrB, as<float>(R_base), device_flag);
            return;
        case 8:
            cpp_vclVector_elem_log_base<double>(ptrA, ptrB, as<double>(R_base), device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclVector_elem_exp(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_exp<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_vclVector_elem_exp<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_vclVector_elem_exp<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}
