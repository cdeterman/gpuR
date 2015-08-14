
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

template <typename T>
void cpp_vienna_axpy(
    T const alpha, 
    MapMat<T> &Am, 
    MapMat<T> &Bm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
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
void cpp_vienna_vec_axpy(
    T const alpha, 
    MapVec<T> &Am, 
    MapVec<T> &Bm,
    int device_flag)
{    
    if(device_flag == 0){
        //use only GPUs:
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_B(M);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_B += alpha * vcl_A;
    
    viennacl::copy(vcl_B, Bm);
}


template <typename T>
T cpp_vienna_gev_inner(
    MapVec<T> &Am, 
    MapVec<T> &Bm, 
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    T C;
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_B(M);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    C = viennacl::linalg::inner_prod(vcl_A, vcl_B);
    return C;
    
}

template <typename T>
void cpp_vienna_gev_outer(
    MapVec<T> &Am, 
    MapVec<T> &Bm, 
    MapMat<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_B(M);
    viennacl::matrix<T> vcl_C(M, M);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_C = viennacl::linalg::outer_prod(vcl_A, vcl_B);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_vec_elem_prod(
    MapVec<T> &Am, 
    MapVec<T> &Bm, 
    MapVec<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
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
void cpp_vienna_vec_elem_div(
    MapVec<T> &Am, 
    MapVec<T> &Bm, 
    MapVec<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
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
void cpp_vienna_vec_elem_sin(
    MapVec<T> &Am, 
    MapVec<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_sin(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_vec_elem_asin(
    MapVec<T> &Am, 
    MapVec<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_asin(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_vec_elem_sinh(
    MapVec<T> &Am, 
    MapVec<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_sinh(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_vec_elem_cos(
    MapVec<T> &Am, 
    MapVec<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_cos(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_vec_elem_acos(
    MapVec<T> &Am, 
    MapVec<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_acos(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_vec_elem_cosh(
    MapVec<T> &Am, 
    MapVec<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_cosh(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_vec_elem_tan(
    MapVec<T> &Am, 
    MapVec<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_tan(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_vec_elem_atan(
    MapVec<T> &Am, 
    MapVec<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_atan(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_vec_elem_tanh(
    MapVec<T> &Am, 
    MapVec<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M);
    viennacl::vector<T> vcl_C(M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_tanh(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_elem_prod(
    MapMat<T> &Am, 
    MapMat<T> &Bm, 
    MapMat<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
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
    MapMat<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
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
    MapMat<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_sin(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_elem_asin(
    MapMat<T> &Am, 
    MapMat<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_asin(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_elem_sinh(
    MapMat<T> &Am, 
    MapMat<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_sinh(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_elem_cos(
    MapMat<T> &Am, 
    MapMat<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_cos(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_elem_acos(
    MapMat<T> &Am, 
    MapMat<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_acos(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_elem_cosh(
    MapMat<T> &Am, 
    MapMat<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_cosh(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_elem_tan(
    MapMat<T> &Am, 
    MapMat<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_tan(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_elem_atan(
    MapMat<T> &Am, 
    MapMat<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_atan(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_elem_tanh(
    MapMat<T> &Am, 
    MapMat<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_tanh(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_elem_log(
    MapMat<T> &Am, 
    MapMat<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_log(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_elem_log_base(
    MapMat<T> &Am, 
    MapMat<T> &Cm,
    T base,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_log10(vcl_A);
    vcl_C /= log10(base);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_elem_log10(
    MapMat<T> &Am, 
    MapMat<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_log10(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_vienna_elem_exp(
    MapMat<T> &Am, 
    MapMat<T> &Cm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_C(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_exp(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

/*** gpuMatrix functions ***/

//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_daxpy(
    SEXP alpha_, 
    SEXP ptrA_, 
    SEXP ptrB_,
    int device_flag)
{
    const double alpha = as<double>(alpha_);
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrB(ptrB_);
    
    MapMat<double> Am = MapMat<double>(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Bm = MapMat<double>(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    
    cpp_vienna_axpy(alpha, Am, Bm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_saxpy(
    SEXP alpha_, 
    SEXP ptrA_, 
    SEXP ptrB_,
    int device_flag)
{
    const float alpha = as<float>(alpha_);

    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrB(ptrB_);
    
    MapMat<float> Am = MapMat<float>(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Bm = MapMat<float>(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    
    cpp_vienna_axpy(alpha, Am, Bm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_elem_prod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_prod<double>(Am, Bm, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuMatrix_elem_prod(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_elem_prod<float>(Am, Bm, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_elem_div(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_div<double>(Am, Bm, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuMatrix_elem_div(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_elem_div<float>(Am, Bm, Cm, device_flag);
}


//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_elem_sin(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_sin<double>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_elem_asin(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_asin<double>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_elem_sinh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_sinh<double>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuMatrix_elem_sin(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_elem_sin<float>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuMatrix_elem_asin(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_elem_asin<float>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuMatrix_elem_sinh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_elem_sinh<float>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_elem_cos(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_cos<double>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_elem_acos(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_acos<double>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_elem_cosh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_cosh<double>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuMatrix_elem_cos(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_elem_cos<float>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuMatrix_elem_acos(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_elem_acos<float>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuMatrix_elem_cosh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_elem_cosh<float>(Am, Cm, device_flag);
}


//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_elem_tan(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_tan<double>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_elem_atan(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_atan<double>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_elem_tanh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_tanh<double>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuMatrix_elem_tan(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_elem_tan<float>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuMatrix_elem_atan(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_elem_atan<float>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuMatrix_elem_tanh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_elem_tanh<float>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuMatrix_elem_log(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_elem_log<float>(Am, Cm, device_flag);
}


//[[Rcpp::export]]
void cpp_vienna_sgpuMatrix_elem_log10(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_elem_log10<float>(Am, Cm, device_flag);
}


//[[Rcpp::export]]
void cpp_vienna_sgpuMatrix_elem_log_base(
    SEXP ptrA_, 
    SEXP ptrC_,
    float base,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_log_base<float>(Am, Cm, base, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuMatrix_elem_exp(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_vienna_elem_exp<float>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_elem_log(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_log<double>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_elem_log10(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_log10<double>(Am, Cm, device_flag);
}


//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_elem_log_base(
    SEXP ptrA_, 
    SEXP ptrC_,
    double base,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_log_base<double>(Am, Cm, base, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_elem_exp(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_elem_exp<double>(Am, Cm, device_flag);
}

/*** vclMatrix Functions ***/

//[[Rcpp::export]]
void cpp_vclMatrix_daxpy(SEXP alpha_, 
                        SEXP ptrA_, 
                        SEXP ptrB_,
                        int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    const double alpha = as<double>(alpha_);
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrB(ptrB_);
    
    *ptrB += alpha * (*ptrA);
}

//[[Rcpp::export]]
void cpp_vclMatrix_saxpy(
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
    
    const float alpha = as<float>(alpha_);

    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrB(ptrB_);
    
    *ptrB += alpha * (*ptrA);
}


//[[Rcpp::export]]
void cpp_dvclMatrix_elem_prod(
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
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_prod(*ptrA, *ptrB);
}

//[[Rcpp::export]]
void cpp_svclMatrix_elem_prod(
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
    
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_prod(*ptrA, *ptrB);
}

//[[Rcpp::export]]
void cpp_dvclMatrix_elem_div(
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
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_div(*ptrA, *ptrB);
}

//[[Rcpp::export]]
void cpp_svclMatrix_elem_div(
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
    
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_div(*ptrA, *ptrB);
}


//[[Rcpp::export]]
void cpp_dvclMatrix_elem_sin(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{ 
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_sin(*ptrA);
}

//[[Rcpp::export]]
void cpp_dvclMatrix_elem_asin(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_asin(*ptrA);
}

//[[Rcpp::export]]
void cpp_dvclMatrix_elem_sinh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_sinh(*ptrA);
}

//[[Rcpp::export]]
void cpp_svclMatrix_elem_sin(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_sin(*ptrA);
}

//[[Rcpp::export]]
void cpp_svclMatrix_elem_asin(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_asin(*ptrA);
}

//[[Rcpp::export]]
void cpp_svclMatrix_elem_sinh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_sinh(*ptrA);
}

//[[Rcpp::export]]
void cpp_dvclMatrix_elem_cos(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_cos(*ptrA);
}

//[[Rcpp::export]]
void cpp_dvclMatrix_elem_acos(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_acos(*ptrA);
}

//[[Rcpp::export]]
void cpp_dvclMatrix_elem_cosh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_cosh(*ptrA);
}

//[[Rcpp::export]]
void cpp_svclMatrix_elem_cos(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_cos(*ptrA);
}

//[[Rcpp::export]]
void cpp_svclMatrix_elem_acos(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_acos(*ptrA);
}

//[[Rcpp::export]]
void cpp_svclMatrix_elem_cosh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_cosh(*ptrA);
}


//[[Rcpp::export]]
void cpp_dvclMatrix_elem_tan(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_tan(*ptrA);
}

//[[Rcpp::export]]
void cpp_dvclMatrix_elem_atan(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_atan(*ptrA);
}

//[[Rcpp::export]]
void cpp_dvclMatrix_elem_tanh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_tanh(*ptrA);
}

//[[Rcpp::export]]
void cpp_svclMatrix_elem_tan(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_tan(*ptrA);
}

//[[Rcpp::export]]
void cpp_svclMatrix_elem_atan(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_atan(*ptrA);
}

//[[Rcpp::export]]
void cpp_svclMatrix_elem_tanh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_tanh(*ptrA);
}

//[[Rcpp::export]]
void cpp_svclMatrix_elem_log(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_log(*ptrA);
}


//[[Rcpp::export]]
void cpp_svclMatrix_elem_log10(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_log10(*ptrA);
}


//[[Rcpp::export]]
void cpp_svclMatrix_elem_log_base(
    SEXP ptrA_, 
    SEXP ptrC_,
    float base,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_log10(*ptrA);
    *ptrC /= log10(base);
}

//[[Rcpp::export]]
void cpp_svclMatrix_elem_exp(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_exp(*ptrA);
}

//[[Rcpp::export]]
void cpp_dvclMatrix_elem_log(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_log(*ptrA);
}

//[[Rcpp::export]]
void cpp_dvclMatrix_elem_log10(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{ 
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_log10(*ptrA);
}


//[[Rcpp::export]]
void cpp_dvclMatrix_elem_log_base(
    SEXP ptrA_, 
    SEXP ptrC_,
    double base,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_log10(*ptrA);
    *ptrC /= log10(base);
}

//[[Rcpp::export]]
void cpp_dvclMatrix_elem_exp(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_exp(*ptrA);
}

/*** gpuVector functions ***/

//[[Rcpp::export]]
SEXP cpp_vienna_gpuVector_dgev_inner(
    SEXP ptrA_, 
    SEXP ptrB_,
    int device_flag)
{
    double out;    
    Rcpp::XPtr<dynEigenVec<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<double> > ptrB(ptrB_);
    
    MapVec<double> Am(ptrA->ptr(), ptrA->length());
    MapVec<double> Bm(ptrB->ptr(), ptrB->length());

    out = cpp_vienna_gev_inner<double>(Am, Bm, device_flag);
    return(Rcpp::wrap(out));
}

//[[Rcpp::export]]
SEXP cpp_vienna_gpuVector_sgev_inner(
    SEXP ptrA_, 
    SEXP ptrB_,
    int device_flag)
{
    float out;
    Rcpp::XPtr<dynEigenVec<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<float> > ptrB(ptrB_);
    
    MapVec<float> Am(ptrA->ptr(), ptrA->length());
    MapVec<float> Bm(ptrB->ptr(), ptrB->length());

    out = cpp_vienna_gev_inner<float>(Am, Bm, device_flag);
    return(Rcpp::wrap(out));
}


//[[Rcpp::export]]
void cpp_vienna_gpuVector_dgev_outer(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_,
    int device_flag)
{
    Rcpp::XPtr<dynEigenVec<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<double> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapVec<double> Am(ptrA->ptr(), ptrA->length());
    MapVec<double> Bm(ptrB->ptr(), ptrB->length());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_gev_outer<double>(Am, Bm, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_gpuVector_sgev_outer(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_,
    int device_flag)
{
    Rcpp::XPtr<dynEigenVec<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<float> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapVec<float> Am(ptrA->ptr(), ptrA->length());
    MapVec<float> Bm(ptrB->ptr(), ptrB->length());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_vienna_gev_outer<float>(Am, Bm, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_gpuVector_daxpy(
    SEXP alpha_, SEXP A_, SEXP B_,
    int device_flag)
{
    const double alpha = as<double>(alpha_);
    Rcpp::XPtr<dynEigenVec<double> > ptrA(A_);
    Rcpp::XPtr<dynEigenVec<double> > ptrB(B_);
    
    MapVec<double> Am(ptrA->ptr(), ptrA->length());
    MapVec<double> Bm(ptrB->ptr(), ptrB->length());
    
    cpp_vienna_vec_axpy<double>(alpha, Am, Bm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_gpuVector_saxpy(
    SEXP alpha_, SEXP A_, SEXP B_,
    int device_flag)
{
    const float alpha = as<float>(alpha_);
    Rcpp::XPtr<dynEigenVec<float> > ptrA(A_);
    Rcpp::XPtr<dynEigenVec<float> > ptrB(B_);
    
    MapVec<float> Am(ptrA->ptr(), ptrA->length());
    MapVec<float> Bm(ptrB->ptr(), ptrB->length());
    
    cpp_vienna_vec_axpy<float>(alpha, Am, Bm, device_flag);
}


//[[Rcpp::export]]
void cpp_vienna_dgpuVector_elem_prod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigenVec<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<double> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigenVec<double> > ptrC(ptrC_);
    
    MapVec<double> Am(ptrA->ptr(), ptrA->length());
    MapVec<double> Bm(ptrB->ptr(), ptrB->length());
    MapVec<double> Cm(ptrC->ptr(), ptrC->length());

    cpp_vienna_vec_elem_prod<double>(Am, Bm, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuVector_elem_prod(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigenVec<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<float> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigenVec<float> > ptrC(ptrC_);
    
    MapVec<float> Am(ptrA->ptr(), ptrA->length());
    MapVec<float> Bm(ptrB->ptr(), ptrB->length());
    MapVec<float> Cm(ptrC->ptr(), ptrC->length());
    
    cpp_vienna_vec_elem_prod<float>(Am, Bm, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuVector_elem_div(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigenVec<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<double> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigenVec<double> > ptrC(ptrC_);
    
    MapVec<double> Am(ptrA->ptr(), ptrA->length());
    MapVec<double> Bm(ptrB->ptr(), ptrB->length());
    MapVec<double> Cm(ptrC->ptr(), ptrC->length());

    cpp_vienna_vec_elem_div<double>(Am, Bm, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuVector_elem_div(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigenVec<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<float> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigenVec<float> > ptrC(ptrC_);
    
    MapVec<float> Am(ptrA->ptr(), ptrA->length());
    MapVec<float> Bm(ptrB->ptr(), ptrB->length());
    MapVec<float> Cm(ptrC->ptr(), ptrC->length());
    
    cpp_vienna_vec_elem_div<float>(Am, Bm, Cm, device_flag);
}


//[[Rcpp::export]]
void cpp_vienna_dgpuVector_elem_sin(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigenVec<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<double> > ptrC(ptrC_);
    
    MapVec<double> Am(ptrA->ptr(), ptrA->length());
    MapVec<double> Cm(ptrC->ptr(), ptrC->length());

    cpp_vienna_vec_elem_sin<double>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuVector_elem_asin(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigenVec<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<double> > ptrC(ptrC_);
    
    MapVec<double> Am(ptrA->ptr(), ptrA->length());
    MapVec<double> Cm(ptrC->ptr(), ptrC->length());

    cpp_vienna_vec_elem_asin<double>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuVector_elem_sinh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigenVec<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<double> > ptrC(ptrC_);
    
    MapVec<double> Am(ptrA->ptr(), ptrA->length());
    MapVec<double> Cm(ptrC->ptr(), ptrC->length());

    cpp_vienna_vec_elem_sinh<double>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuVector_elem_sin(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigenVec<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<float> > ptrC(ptrC_);
    
    MapVec<float> Am(ptrA->ptr(), ptrA->length());
    MapVec<float> Cm(ptrC->ptr(), ptrC->length());
    
    cpp_vienna_vec_elem_sin<float>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuVector_elem_asin(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigenVec<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<float> > ptrC(ptrC_);
    
    MapVec<float> Am(ptrA->ptr(), ptrA->length());
    MapVec<float> Cm(ptrC->ptr(), ptrC->length());
    
    cpp_vienna_vec_elem_asin<float>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuVector_elem_sinh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigenVec<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<float> > ptrC(ptrC_);
    
    MapVec<float> Am(ptrA->ptr(), ptrA->length());
    MapVec<float> Cm(ptrC->ptr(), ptrC->length());
    
    cpp_vienna_vec_elem_sinh<float>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuVector_elem_cos(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigenVec<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<double> > ptrC(ptrC_);
    
    MapVec<double> Am(ptrA->ptr(), ptrA->length());
    MapVec<double> Cm(ptrC->ptr(), ptrC->length());

    cpp_vienna_vec_elem_cos<double>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuVector_elem_acos(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigenVec<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<double> > ptrC(ptrC_);
    
    MapVec<double> Am(ptrA->ptr(), ptrA->length());
    MapVec<double> Cm(ptrC->ptr(), ptrC->length());

    cpp_vienna_vec_elem_acos<double>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuVector_elem_cosh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigenVec<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<double> > ptrC(ptrC_);
    
    MapVec<double> Am(ptrA->ptr(), ptrA->length());
    MapVec<double> Cm(ptrC->ptr(), ptrC->length());

    cpp_vienna_vec_elem_cosh<double>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuVector_elem_cos(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigenVec<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<float> > ptrC(ptrC_);
    
    MapVec<float> Am(ptrA->ptr(), ptrA->length());
    MapVec<float> Cm(ptrC->ptr(), ptrC->length());
    
    cpp_vienna_vec_elem_cos<float>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuVector_elem_acos(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigenVec<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<float> > ptrC(ptrC_);
    
    MapVec<float> Am(ptrA->ptr(), ptrA->length());
    MapVec<float> Cm(ptrC->ptr(), ptrC->length());
    
    cpp_vienna_vec_elem_acos<float>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuVector_elem_cosh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigenVec<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<float> > ptrC(ptrC_);
    
    MapVec<float> Am(ptrA->ptr(), ptrA->length());
    MapVec<float> Cm(ptrC->ptr(), ptrC->length());
    
    cpp_vienna_vec_elem_cosh<float>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuVector_elem_tan(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigenVec<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<double> > ptrC(ptrC_);
    
    MapVec<double> Am(ptrA->ptr(), ptrA->length());
    MapVec<double> Cm(ptrC->ptr(), ptrC->length());

    cpp_vienna_vec_elem_tan<double>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuVector_elem_atan(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigenVec<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<double> > ptrC(ptrC_);
    
    MapVec<double> Am(ptrA->ptr(), ptrA->length());
    MapVec<double> Cm(ptrC->ptr(), ptrC->length());

    cpp_vienna_vec_elem_atan<double>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuVector_elem_tanh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigenVec<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<double> > ptrC(ptrC_);
    
    MapVec<double> Am(ptrA->ptr(), ptrA->length());
    MapVec<double> Cm(ptrC->ptr(), ptrC->length());

    cpp_vienna_vec_elem_tanh<double>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuVector_elem_tan(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigenVec<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<float> > ptrC(ptrC_);
    
    MapVec<float> Am(ptrA->ptr(), ptrA->length());
    MapVec<float> Cm(ptrC->ptr(), ptrC->length());
    
    cpp_vienna_vec_elem_tan<float>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuVector_elem_atan(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigenVec<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<float> > ptrC(ptrC_);
    
    MapVec<float> Am(ptrA->ptr(), ptrA->length());
    MapVec<float> Cm(ptrC->ptr(), ptrC->length());
    
    cpp_vienna_vec_elem_atan<float>(Am, Cm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_sgpuVector_elem_tanh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    Rcpp::XPtr<dynEigenVec<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<float> > ptrC(ptrC_);
    
    MapVec<float> Am(ptrA->ptr(), ptrA->length());
    MapVec<float> Cm(ptrC->ptr(), ptrC->length());
    
    cpp_vienna_vec_elem_tanh<float>(Am, Cm, device_flag);
}


/*** vclVector Functions ***/

//[[Rcpp::export]]
void cpp_vclVector_daxpy(
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
    
    const double alpha = as<double>(alpha_);
    
    Rcpp::XPtr<viennacl::vector<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<double> > ptrB(ptrB_);
    
    *ptrB += alpha * (*ptrA);
}

//[[Rcpp::export]]
void cpp_vclVector_saxpy(
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
    
    const float alpha = as<float>(alpha_);

    Rcpp::XPtr<viennacl::vector<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<float> > ptrB(ptrB_);
    
    *ptrB += alpha * (*ptrA);
}

//[[Rcpp::export]]
double cpp_dvclVector_inner_prod(
    SEXP ptrA_, 
    SEXP ptrB_,
    int device_flag)
{
    double out;
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<double> > ptrB(ptrB_);

    out = viennacl::linalg::inner_prod(*ptrA, *ptrB);
    return out;
}

//[[Rcpp::export]]
float cpp_svclVector_inner_prod(
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
    
    Rcpp::XPtr<viennacl::vector<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<float> > ptrB(ptrB_);
    
    out = viennacl::linalg::inner_prod(*ptrA, *ptrB);
    return out;
}

//[[Rcpp::export]]
void cpp_dvclVector_outer_prod(
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
    
    Rcpp::XPtr<viennacl::vector<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<double> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::outer_prod(*ptrA, *ptrB);
}

//[[Rcpp::export]]
void cpp_svclVector_outer_prod(
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
    
    Rcpp::XPtr<viennacl::vector<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<float> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::outer_prod(*ptrA, *ptrB);
}

//[[Rcpp::export]]
void cpp_dvclVector_elem_prod(
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
    
    Rcpp::XPtr<viennacl::vector<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<double> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::vector<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_prod(*ptrA, *ptrB);
}

//[[Rcpp::export]]
void cpp_svclVector_elem_prod(
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
    
    Rcpp::XPtr<viennacl::vector<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<float> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::vector<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_prod(*ptrA, *ptrB);
}

//[[Rcpp::export]]
void cpp_dvclVector_elem_div(
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
    
    Rcpp::XPtr<viennacl::vector<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<double> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::vector<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_div(*ptrA, *ptrB);
}

//[[Rcpp::export]]
void cpp_svclVector_elem_div(
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
    
    Rcpp::XPtr<viennacl::vector<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<float> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::vector<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_div(*ptrA, *ptrB);
}


//[[Rcpp::export]]
void cpp_dvclVector_elem_sin(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{ 
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_sin(*ptrA);
}

//[[Rcpp::export]]
void cpp_dvclVector_elem_asin(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_asin(*ptrA);
}

//[[Rcpp::export]]
void cpp_dvclVector_elem_sinh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_sinh(*ptrA);
}

//[[Rcpp::export]]
void cpp_svclVector_elem_sin(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_sin(*ptrA);
}

//[[Rcpp::export]]
void cpp_svclVector_elem_asin(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_asin(*ptrA);
}

//[[Rcpp::export]]
void cpp_svclVector_elem_sinh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_sinh(*ptrA);
}

//[[Rcpp::export]]
void cpp_dvclVector_elem_cos(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_cos(*ptrA);
}

//[[Rcpp::export]]
void cpp_dvclVector_elem_acos(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_acos(*ptrA);
}

//[[Rcpp::export]]
void cpp_dvclVector_elem_cosh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_cosh(*ptrA);
}

//[[Rcpp::export]]
void cpp_svclVector_elem_cos(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_cos(*ptrA);
}

//[[Rcpp::export]]
void cpp_svclVector_elem_acos(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_acos(*ptrA);
}

//[[Rcpp::export]]
void cpp_svclVector_elem_cosh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_cosh(*ptrA);
}


//[[Rcpp::export]]
void cpp_dvclVector_elem_tan(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_tan(*ptrA);
}

//[[Rcpp::export]]
void cpp_dvclVector_elem_atan(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_atan(*ptrA);
}

//[[Rcpp::export]]
void cpp_dvclVector_elem_tanh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_tanh(*ptrA);
}

//[[Rcpp::export]]
void cpp_svclVector_elem_tan(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_tan(*ptrA);
}

//[[Rcpp::export]]
void cpp_svclVector_elem_atan(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_atan(*ptrA);
}

//[[Rcpp::export]]
void cpp_svclVector_elem_tanh(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_tanh(*ptrA);
}

//[[Rcpp::export]]
void cpp_svclVector_elem_log(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_log(*ptrA);
}


//[[Rcpp::export]]
void cpp_svclVector_elem_log10(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_log10(*ptrA);
}


//[[Rcpp::export]]
void cpp_svclVector_elem_log_base(
    SEXP ptrA_, 
    SEXP ptrC_,
    float base,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_log10(*ptrA);
    *ptrC /= log10(base);
}

//[[Rcpp::export]]
void cpp_svclVector_elem_exp(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<float> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_exp(*ptrA);
}

//[[Rcpp::export]]
void cpp_dvclVector_elem_log(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_log(*ptrA);
}

//[[Rcpp::export]]
void cpp_dvclVector_elem_log10(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{ 
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_log10(*ptrA);
}


//[[Rcpp::export]]
void cpp_dvclVector_elem_log_base(
    SEXP ptrA_, 
    SEXP ptrC_,
    double base,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_log10(*ptrA);
    *ptrC /= log10(base);
}

//[[Rcpp::export]]
void cpp_dvclVector_elem_exp(
    SEXP ptrA_, 
    SEXP ptrC_,
    int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::vector<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::vector<double> > ptrC(ptrC_);

    *ptrC = viennacl::linalg::element_exp(*ptrA);
}
