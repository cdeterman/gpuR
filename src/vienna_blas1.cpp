
#include "gpuR/windows_check.hpp"

// eigen headers for handling the R input data
#include <RcppEigen.h>

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
#include "viennacl/linalg/maxmin.hpp"

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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(A_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrB(B_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Bm(ptrB->data(), ptrB->rows(), 1);
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_B(M);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_B += alpha * vcl_A;
    
    viennacl::copy(vcl_B, Bm);
}

template <typename T>
void 
cpp_gpuVector_unary_axpy(
    SEXP ptrA_, 
    const int device_flag)
{
         //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_Z = viennacl::zero_vector<T>(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_Z -= vcl_A;

    viennacl::copy(vcl_Z, Am);
}

//template <typename T>
//void cpp_gpuSlicedVector_axpy(
//    SEXP alpha_, 
//    SEXP A_, SEXP B_,
//    int device_flag)
//{
//    if(device_flag == 0){
//        //use only GPUs:
//        long id = 0;
//        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
//    }
//    
//    const T alpha = as<T>(alpha_);
////    Rcpp::XPtr<dynEigenVec<T> > ptrA(A_);
////    Rcpp::XPtr<dynEigenVec<T> > ptrB(B_);
////    
////    Eigen::Matrix<T, Eigen::Dynamic, 1> Am(ptrA->end() - ptrA->start());
////    Eigen::Matrix<T, Eigen::Dynamic, 1> Bm(ptrA->end() - ptrA->start());
//    
//    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
//    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrB(ptrB_);
//    
//    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
//    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Bm(ptrB->data(), ptrB->rows(), 1);
//    
//    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > tempA(ptrA->ptr(), ptrA->length());
//    // a copy
//    Am = tempA.segment(ptrA->start(), ptrA->end());
//    
//    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > tempB(ptrB->ptr(), ptrB->length());
//    // a copy
//    Bm = tempB.segment(ptrB->start(), ptrB->end());
//    
//    int M = Am.size();
//    
//    viennacl::vector<T> vcl_A(M);
//    viennacl::vector<T> vcl_B(M);
//    
//    viennacl::copy(Am, vcl_A); 
//    viennacl::copy(Bm, vcl_B); 
//    
//    vcl_B += alpha * vcl_A;
//    
//    viennacl::copy(vcl_B, Bm);
//}

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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Bm(ptrB->data(), ptrB->rows(), 1);
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrB(ptrB_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Bm(ptrB->data(), ptrB->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Cm(ptrC->data(), ptrC->rows(), ptrC->cols());
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrB(ptrB_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Bm(ptrB->data(), ptrB->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm(ptrC->data(), ptrC->rows(), 1);
    
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
void 
cpp_gpuVector_scalar_prod(
    SEXP ptrC_, 
    SEXP scalar, 
    const int device_flag)
{        
    const T alpha = as<T>(scalar);
    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrC(ptrC_);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm(ptrC->data(), ptrC->rows(), 1);
    
    int M = Cm.size();
    
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Cm, vcl_C); 
    
    vcl_C *= alpha;
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrB(ptrB_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Bm(ptrB->data(), ptrB->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm(ptrC->data(), ptrC->rows(), 1);
    
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
cpp_gpuVector_scalar_div(
    SEXP ptrC_, 
    SEXP scalar, 
    const int order,
    const int device_flag)
{        
    const T alpha = as<T>(scalar);
    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrC(ptrC_);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm(ptrC->data(), ptrC->rows(), 1);
    
    int M = Cm.size();
    
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Cm, vcl_C); 
    
    if(order == 0){
        vcl_C /= alpha;
        viennacl::copy(vcl_C, Cm);
    }else{
        viennacl::vector<T> vcl_scalar = viennacl::scalar_vector<T>(M, alpha);
        vcl_scalar = viennacl::linalg::element_div(vcl_scalar, vcl_C);
        viennacl::copy(vcl_scalar, Cm);
    }
}

template <typename T>
void cpp_gpuVector_elem_pow(
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrB(ptrB_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Bm(ptrB->data(), ptrB->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm(ptrC->data(), ptrC->rows(), 1);
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_B(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_C = viennacl::linalg::element_pow(vcl_A, vcl_B);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_gpuVector_scalar_pow(
    SEXP ptrA_, 
    SEXP scalar_, 
    SEXP ptrC_,
    const int order,
    const int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    const T scalar = as<T>(scalar_);    
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm(ptrC->data(), ptrC->rows(), 1);
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    viennacl::vector<T> vcl_B = viennacl::scalar_vector<T>(M, scalar);
    
    viennacl::copy(Am, vcl_A); 
    
    if(order == 0){
        vcl_C = viennacl::linalg::element_pow(vcl_A, vcl_B);
    }else{
        vcl_C = viennacl::linalg::element_pow(vcl_B, vcl_A);
    }
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm(ptrC->data(), ptrC->rows(), 1);
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm(ptrC->data(), ptrC->rows(), 1);
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm(ptrC->data(), ptrC->rows(), 1);
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm(ptrC->data(), ptrC->rows(), 1);
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm(ptrC->data(), ptrC->rows(), 1);
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm(ptrC->data(), ptrC->rows(), 1);
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm(ptrC->data(), ptrC->rows(), 1);
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm(ptrC->data(), ptrC->rows(), 1);
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm(ptrC->data(), ptrC->rows(), 1);
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_tanh(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_exp(
    SEXP ptrA_, SEXP ptrC_,
    const int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm(ptrC->data(), ptrC->rows(), 1);
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_exp(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_log10(
    SEXP ptrA_, SEXP ptrC_,
    const int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm(ptrC->data(), ptrC->rows(), 1);
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_log10(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_log(
    SEXP ptrA_, SEXP ptrC_,
    const int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm(ptrC->data(), ptrC->rows(), 1);
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_log(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_log_base(
    SEXP ptrA_, SEXP ptrB_,
    T base,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Bm(ptrB->data(), ptrB->rows(), 1);
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_B(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_B = viennacl::linalg::element_log10(vcl_A);
    vcl_B /= log10(base);
    
    viennacl::copy(vcl_B, Bm);
}

template <typename T>
void 
cpp_gpuVector_elem_abs(
    SEXP ptrA_, SEXP ptrC_,
    const int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm(ptrC->data(), ptrC->rows(), 1);
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_fabs(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
T
cpp_gpuVector_max(
    SEXP ptrA_,
    const int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    T max;
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    
    viennacl::copy(Am, vcl_A); 
    
    max = viennacl::linalg::max(vcl_A);
    
    return max;
}

template <typename T>
T
cpp_gpuVector_min(
    SEXP ptrA_,
    const int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    T max;
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptrA(ptrA_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am(ptrA->data(), ptrA->rows(), 1);
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    
    viennacl::copy(Am, vcl_A); 
    
    max = viennacl::linalg::min(vcl_A);
    
    return max;
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->data(), ptrB->rows(), ptrB->cols());
    
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
void 
cpp_gpuMatrix_unary_axpy(
    SEXP ptrA_, 
    const int device_flag)
{
         //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
    
    const int M = Am.cols();
    const int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_Z = viennacl::zero_matrix<T>(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_Z -= vcl_A;

    viennacl::copy(vcl_Z, Am);
}

template <typename T>
void 
cpp_gpuMatrix_elem_prod(
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
        
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrB(ptrB_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->data(), ptrB->rows(), ptrB->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Cm(ptrC->data(), ptrC->rows(), ptrC->cols());
    
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
void 
cpp_gpuMatrix_scalar_prod(
    SEXP ptrC_, 
    SEXP scalar, 
    const int device_flag)
{        
    const T alpha = as<T>(scalar);
    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Cm(ptrC->data(), ptrC->rows(), ptrC->cols());
    
    int M = Cm.cols();
    int K = Cm.rows();
    
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Cm, vcl_C); 
    
    vcl_C *= alpha;
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuMatrix_elem_div(
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrB(ptrB_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->data(), ptrB->rows(), ptrB->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Cm(ptrC->data(), ptrC->rows(), ptrC->cols());
    
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
void 
cpp_gpuMatrix_scalar_div(
    SEXP ptrC_, 
    SEXP B_scalar, 
    const int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    T B = Rcpp::as<T>(B_scalar);
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Cm(ptrC->data(), ptrC->rows(), ptrC->cols());
    
    int M = Cm.cols();
    int K = Cm.rows();
    
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Cm, vcl_C); 
    
    vcl_C /= B;
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuMatrix_elem_pow(
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrB(ptrB_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->data(), ptrB->rows(), ptrB->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Cm(ptrC->data(), ptrC->rows(), ptrC->cols());
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(K,M);
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_C = viennacl::linalg::element_pow(vcl_A, vcl_B);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuMatrix_scalar_pow(
    SEXP ptrA_, 
    SEXP scalar_, 
    SEXP ptrC_, 
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    const T scalar = as<T>(scalar_);
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Cm(ptrC->data(), ptrC->rows(), ptrC->cols());
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_C(K,M);
    viennacl::matrix<T> vcl_B = viennacl::scalar_matrix<T>(K,M,scalar);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_pow(vcl_A, vcl_B);
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->data(), ptrB->rows(), ptrB->cols());
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->data(), ptrB->rows(), ptrB->cols());
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->data(), ptrB->rows(), ptrB->cols());
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->data(), ptrB->rows(), ptrB->cols());
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->data(), ptrB->rows(), ptrB->cols());
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->data(), ptrB->rows(), ptrB->cols());
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->data(), ptrB->rows(), ptrB->cols());
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->data(), ptrB->rows(), ptrB->cols());
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->data(), ptrB->rows(), ptrB->cols());
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->data(), ptrB->rows(), ptrB->cols());
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->data(), ptrB->rows(), ptrB->cols());
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->data(), ptrB->rows(), ptrB->cols());
    
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
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->data(), ptrB->rows(), ptrB->cols());
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_B = viennacl::linalg::element_exp(vcl_A);
    
    viennacl::copy(vcl_B, Bm);
}

template <typename T>
void cpp_gpuMatrix_elem_abs(
    SEXP ptrA_, SEXP ptrB_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
    XPtr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->data(), ptrB->rows(), ptrB->cols());
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_B = viennacl::linalg::element_fabs(vcl_A);
    
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
cpp_gpuMatrix_scalar_prod(
    SEXP ptrC,
    SEXP scalar,
    const int device_flag,
    const int type_flag)
{    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_scalar_prod<int>(ptrC, scalar, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_scalar_prod<float>(ptrC, scalar, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_scalar_prod<double>(ptrC, scalar, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_scalar_div(
    SEXP ptrC,
    SEXP B_scalar,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_scalar_div<int>(ptrC, B_scalar, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_scalar_div<float>(ptrC, B_scalar, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_scalar_div<double>(ptrC, B_scalar, device_flag);
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
cpp_gpuMatrix_elem_pow(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_pow<int>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_elem_pow<float>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_elem_pow<double>(ptrA, ptrB, ptrC, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_scalar_pow(
    SEXP ptrA, SEXP scalar, SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_scalar_pow<int>(ptrA, scalar, ptrC, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_scalar_pow<float>(ptrA, scalar, ptrC, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_scalar_pow<double>(ptrA, scalar, ptrC, device_flag);
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
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
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
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
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
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
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
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_abs(
    SEXP ptrA, SEXP ptrB,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_abs<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_elem_abs<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_elem_abs<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
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
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}
    
// [[Rcpp::export]]
void
cpp_gpuMatrix_unary_axpy(
    SEXP ptrA,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_unary_axpy<int>(ptrA, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_unary_axpy<float>(ptrA, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_unary_axpy<double>(ptrA, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
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
void cpp_vclVector_elem_pow(
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

    *ptrC = viennacl::linalg::element_pow(*ptrA, *ptrB);
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
void cpp_vclMatrix_elem_pow(
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

    *ptrC = viennacl::linalg::element_pow(*ptrA, *ptrB);
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
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
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
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
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
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

//[[Rcpp::export]]
void cpp_vclMatrix_elem_pow(
    SEXP ptrA, 
    SEXP ptrB,
    SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_pow<int>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 6:
            cpp_vclMatrix_elem_pow<float>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 8:
            cpp_vclMatrix_elem_pow<double>(ptrA, ptrB, ptrC, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
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
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
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
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
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
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
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
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
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
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
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
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
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
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
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
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
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
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
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
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
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
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
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
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
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
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
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
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_unary_axpy(
    SEXP ptrA,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_unary_axpy<int>(ptrA, device_flag);
            return;
        case 6:
            cpp_gpuVector_unary_axpy<float>(ptrA, device_flag);
            return;
        case 8:
            cpp_gpuVector_unary_axpy<double>(ptrA, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
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
            throw Rcpp::exception("unknown type detected for gpuVector object!");
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
            throw Rcpp::exception("unknown type detected for gpuVector object!");
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
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}
    
// [[Rcpp::export]]
void
cpp_gpuVector_scalar_prod(
    SEXP ptrC,
    SEXP scalar,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_scalar_prod<int>(ptrC, scalar, device_flag);
            return;
        case 6:
            cpp_gpuVector_scalar_prod<float>(ptrC, scalar, device_flag);
            return;
        case 8:
            cpp_gpuVector_scalar_prod<double>(ptrC, scalar, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
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
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_scalar_div(
    SEXP ptrC,
    SEXP scalar,
    const int order,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_scalar_div<int>(ptrC, scalar, order, device_flag);
            return;
        case 6:
            cpp_gpuVector_scalar_div<float>(ptrC, scalar, order, device_flag);
            return;
        case 8:
            cpp_gpuVector_scalar_div<double>(ptrC, scalar, order, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_pow(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_pow<int>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 6:
            cpp_gpuVector_elem_pow<float>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 8:
            cpp_gpuVector_elem_pow<double>(ptrA, ptrB, ptrC, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_scalar_pow(
    SEXP ptrA, SEXP scalar, SEXP ptrC,
    const int order,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_scalar_pow<int>(ptrA, scalar, ptrC, order, device_flag);
            return;
        case 6:
            cpp_gpuVector_scalar_pow<float>(ptrA, scalar, ptrC, order, device_flag);
            return;
        case 8:
            cpp_gpuVector_scalar_pow<double>(ptrA, scalar, ptrC, order, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
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
            throw Rcpp::exception("unknown type detected for gpuVector object!");
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
            throw Rcpp::exception("unknown type detected for gpuVector object!");
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
            throw Rcpp::exception("unknown type detected for gpuVector object!");
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
            throw Rcpp::exception("unknown type detected for gpuVector object!");
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
            throw Rcpp::exception("unknown type detected for gpuVector object!");
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
            throw Rcpp::exception("unknown type detected for gpuVector object!");
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
            throw Rcpp::exception("unknown type detected for gpuVector object!");
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
            throw Rcpp::exception("unknown type detected for gpuVector object!");
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
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_log10(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_log10<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuVector_elem_log10<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuVector_elem_log10<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_log(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_log<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuVector_elem_log<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuVector_elem_log<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_log_base(
    SEXP ptrA, SEXP ptrB,
    SEXP base,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_log_base<int>(ptrA, ptrB, as<int>(base), device_flag);
            return;
        case 6:
            cpp_gpuVector_elem_log_base<float>(ptrA, ptrB, as<float>(base), device_flag);
            return;
        case 8:
            cpp_gpuVector_elem_log_base<double>(ptrA, ptrB, as<double>(base), device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_exp(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_exp<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuVector_elem_exp<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuVector_elem_exp<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_abs(
    SEXP ptrA, SEXP ptrB, 
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_abs<int>(ptrA, ptrB, device_flag);
            return;
        case 6:
            cpp_gpuVector_elem_abs<float>(ptrA, ptrB, device_flag);
            return;
        case 8:
            cpp_gpuVector_elem_abs<double>(ptrA, ptrB, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
SEXP
cpp_gpuVector_max(
    SEXP ptrA,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            return wrap(cpp_gpuVector_max<int>(ptrA, device_flag));
        case 6:
            return wrap(cpp_gpuVector_max<float>(ptrA, device_flag));
        case 8:
            return wrap(cpp_gpuVector_max<double>(ptrA, device_flag));
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
SEXP
cpp_gpuVector_min(
    SEXP ptrA,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            return wrap(cpp_gpuVector_min<int>(ptrA, device_flag));
        case 6:
            return wrap(cpp_gpuVector_min<float>(ptrA, device_flag));
        case 8:
            return wrap(cpp_gpuVector_min<double>(ptrA, device_flag));
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
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
cpp_vclVector_elem_pow(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_pow<int>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 6:
            cpp_vclVector_elem_pow<float>(ptrA, ptrB, ptrC, device_flag);
            return;
        case 8:
            cpp_vclVector_elem_pow<double>(ptrA, ptrB, ptrC, device_flag);
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
