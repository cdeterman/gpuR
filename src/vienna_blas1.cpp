
#include "gpuR/windows_check.hpp"

// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/dynEigenMat.hpp"
#include "gpuR/dynEigenVec.hpp"
#include "gpuR/dynVCLMat.hpp"
#include "gpuR/dynVCLVec.hpp"

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
    int ctx_id)
{
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    const T alpha = as<T>(alpha_);

    XPtr<dynEigenVec<T> > ptrA(A_);
    XPtr<dynEigenVec<T> > ptrB(B_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Bm = ptrB->data();
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_B(M, ctx = ctx);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_B += alpha * vcl_A;
    
    viennacl::copy(vcl_B, Bm);
}

template <typename T>
void 
cpp_gpuVector_unary_axpy(
    SEXP ptrA_,
    int ctx_id)
{
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));

    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_Z = viennacl::zero_vector<T>(M, ctx = ctx);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_Z -= vcl_A;

    viennacl::copy(vcl_Z, Am);
}


template <typename T>
T cpp_gpuVector_inner_prod(
    SEXP ptrA_, 
    SEXP ptrB_,
    int ctx_id)
{   
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    T C;    
    
    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    XPtr<dynEigenVec<T> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Bm = ptrB->data();
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_B(M, ctx = ctx);
    
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
    int ctx_id)
{   
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrC(ptrC_);
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > refC = ptrC->data();
    
    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    XPtr<dynEigenVec<T> > ptrB(ptrB_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Bm = ptrB->data();
//    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Cm(ptrC->data(), ptrC->rows(), ptrC->cols());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Cm(refC.data(), ptrC->nrow(), ptrC->ncol());
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_B(M, ctx = ctx);
    viennacl::matrix<T> vcl_C(M, M, ctx = ctx);
    
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
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));

    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    XPtr<dynEigenVec<T> > ptrB(ptrB_);
    XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Bm = ptrB->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm = ptrC->data();
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_B(M, ctx = ctx);
    viennacl::vector<T> vcl_C(M, ctx = ctx);
    
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
    int ctx_id)
{        
    const T alpha = as<T>(scalar);
    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm = ptrC->data();
    
    int M = Cm.size();
    
    viennacl::vector<T> vcl_C(M, ctx = ctx);
    
    viennacl::copy(Cm, vcl_C); 
    
    vcl_C *= alpha;
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void cpp_gpuVector_elem_div(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    XPtr<dynEigenVec<T> > ptrB(ptrB_);
    XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Bm = ptrB->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm = ptrC->data();
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_B(M, ctx = ctx);
    viennacl::vector<T> vcl_C(M, ctx = ctx);
    
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
    int ctx_id)
{        
    const T alpha = as<T>(scalar);
    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm = ptrC->data();
    
    int M = Cm.size();
    
    viennacl::vector<T> vcl_C(M, ctx = ctx);
    
    viennacl::copy(Cm, vcl_C); 
    
    if(order == 0){
        vcl_C /= alpha;
        viennacl::copy(vcl_C, Cm);
    }else{
        viennacl::vector<T> vcl_scalar = viennacl::scalar_vector<T>(M, alpha, ctx = ctx);
        vcl_scalar = viennacl::linalg::element_div(vcl_scalar, vcl_C);
        viennacl::copy(vcl_scalar, Cm);
    }
}

template <typename T>
void cpp_gpuVector_elem_pow(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_,
    int ctx_id)
{   
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    XPtr<dynEigenVec<T> > ptrB(ptrB_);
    XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Bm = ptrB->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm = ptrC->data();
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_B(M, ctx = ctx);
    viennacl::vector<T> vcl_C(M, ctx = ctx);
    
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
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    const T scalar = as<T>(scalar_);    
    
    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm = ptrC->data();
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_C(M, ctx = ctx);
    viennacl::vector<T> vcl_B = viennacl::scalar_vector<T>(M, scalar, ctx = ctx);
    
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
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm = ptrC->data();
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_C(M, ctx = ctx);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_sin(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_asin(
    SEXP ptrA_, SEXP ptrC_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm = ptrC->data();
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_C(M, ctx = ctx);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_asin(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_sinh(
    SEXP ptrA_, SEXP ptrC_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm = ptrC->data();
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_C(M, ctx = ctx);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_sinh(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_cos(
    SEXP ptrA_, SEXP ptrC_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));

    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm = ptrC->data();
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_C(M, ctx = ctx);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_cos(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_acos(
    SEXP ptrA_, SEXP ptrC_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));

    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm = ptrC->data();
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_C(M, ctx = ctx);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_acos(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_cosh(
    SEXP ptrA_, SEXP ptrC_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm = ptrC->data();const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_C(M, ctx = ctx);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_cosh(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_tan(
    SEXP ptrA_, SEXP ptrC_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm = ptrC->data();
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_C(M, ctx = ctx);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_tan(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_atan(
    SEXP ptrA_, SEXP ptrC_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm = ptrC->data();
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_C(M, ctx = ctx);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_atan(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_tanh(
    SEXP ptrA_, SEXP ptrC_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm = ptrC->data();
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_C(M, ctx = ctx);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_tanh(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_exp(
    SEXP ptrA_, SEXP ptrC_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));

    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm = ptrC->data();
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_C(M, ctx = ctx);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_exp(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_log10(
    SEXP ptrA_, SEXP ptrC_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm = ptrC->data();
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_C(M, ctx = ctx);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_log10(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_log(
    SEXP ptrA_, SEXP ptrC_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm = ptrC->data();
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_C(M, ctx = ctx);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_log(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_log_base(
    SEXP ptrA_, SEXP ptrC_,
    T base,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm = ptrC->data();
    
    int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_C(M, ctx = ctx);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_log10(vcl_A);
    vcl_C /= log10(base);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
void 
cpp_gpuVector_elem_abs(
    SEXP ptrA_, SEXP ptrC_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));

    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    XPtr<dynEigenVec<T> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Cm = ptrC->data();
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    viennacl::vector<T> vcl_C(M, ctx = ctx);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_C = viennacl::linalg::element_fabs(vcl_A);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
T
cpp_gpuVector_max(
    SEXP ptrA_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    T max;
    
    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    
    viennacl::copy(Am, vcl_A); 
    
    max = viennacl::linalg::max(vcl_A);
    
    return max;
}

template <typename T>
T
cpp_gpuVector_min(
    SEXP ptrA_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    T max;

    XPtr<dynEigenVec<T> > ptrA(ptrA_);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > Am = ptrA->data();
    
    const int M = Am.size();
    
    viennacl::vector<T> vcl_A(M, ctx = ctx);
    
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
    int ctx_id)
{
    const T alpha = as<T>(alpha_);
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B = ptrB->device_data(ctx_id);
    
    vcl_B += alpha * vcl_A;

    ptrB->to_host(vcl_B);
}

template <typename T>
void 
cpp_gpuMatrix_unary_axpy(
    SEXP ptrA_,
    int ctx_id)
{
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    
    const int M = ptrA->nrow();
    const int K = ptrA->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_Z = viennacl::zero_matrix<T>(M,K, ctx);
    
    vcl_Z -= vcl_A;

    ptrA->to_host(vcl_Z);
}

template <typename T>
void 
cpp_gpuMatrix_elem_prod(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    XPtr<dynEigenMat<T> > ptrC(ptrC_);
    
    const int K = ptrC->nrow();
    const int M = ptrC->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B = ptrB->device_data(ctx_id);
    viennacl::matrix<T> vcl_C(K,M, ctx = ctx);
    
    vcl_C = viennacl::linalg::element_prod(vcl_A, vcl_B);
    
    ptrC->to_host(vcl_C);
}

template <typename T>
void 
cpp_gpuMatrix_scalar_prod(
    SEXP ptrC_, 
    SEXP scalar,
    int ctx_id)
{        
    const T alpha = as<T>(scalar);

    XPtr<dynEigenMat<T> > ptrC(ptrC_);
    
    viennacl::matrix<T> vcl_C = ptrC->device_data(ctx_id);
    
    vcl_C *= alpha;
    
    ptrC->to_host(vcl_C);
}

template <typename T>
void 
cpp_gpuMatrix_elem_div(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    XPtr<dynEigenMat<T> > ptrC(ptrC_);
    
    const int K = ptrC->nrow();
    const int M = ptrC->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B = ptrB->device_data(ctx_id);
    viennacl::matrix<T> vcl_C(K,M, ctx = ctx);
    
    vcl_C = viennacl::linalg::element_div(vcl_A, vcl_B);
    
    ptrC->to_host(vcl_C);
}

template <typename T>
void 
cpp_gpuMatrix_scalar_div(
    SEXP ptrC_, 
    SEXP B_scalar,
    int ctx_id)
{    
    T B = Rcpp::as<T>(B_scalar);
    
    XPtr<dynEigenMat<T> > ptrC(ptrC_);
    
    viennacl::matrix<T> vcl_C = ptrC->device_data(ctx_id);
    
    vcl_C /= B;
    
    ptrC->to_host(vcl_C);
}

template <typename T>
void 
cpp_gpuMatrix_elem_pow(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    XPtr<dynEigenMat<T> > ptrC(ptrC_);
    
    const int K = ptrC->nrow();
    const int M = ptrC->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B = ptrB->device_data(ctx_id);
    viennacl::matrix<T> vcl_C(K,M, ctx = ctx);
    
    vcl_C = viennacl::linalg::element_pow(vcl_A, vcl_B);
    
    ptrC->to_host(vcl_C);
}

template <typename T>
void 
cpp_gpuMatrix_scalar_pow(
    SEXP ptrA_, 
    SEXP scalar_, 
    SEXP ptrC_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    const T scalar = as<T>(scalar_);    
        
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrC(ptrC_);
    
    const int K = ptrC->nrow();
    const int M = ptrC->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_C(K,M, ctx = ctx);
    
    viennacl::matrix<T> vcl_B = viennacl::scalar_matrix<T>(K,M,scalar, ctx = ctx);
    
    // std::cout << vcl_A << std::endl;
    // std::cout << vcl_B << std::endl;
    
    vcl_C = viennacl::linalg::element_pow(vcl_A, vcl_B);
    
    // std::cout << vcl_C << std::endl;
    
    ptrC->to_host(vcl_C);
}

template <typename T>
void cpp_gpuMatrix_elem_sin(
    SEXP ptrA_, 
    SEXP ptrB_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);    
    
    const int K = ptrB->nrow();
    const int M = ptrB->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B(K,M, ctx = ctx);
    
    vcl_B = viennacl::linalg::element_sin(vcl_A);
    
    ptrB->to_host(vcl_B);
}

template <typename T>
void cpp_gpuMatrix_elem_asin(
    SEXP ptrA_, 
    SEXP ptrB_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    
    const int K = ptrB->nrow();
    const int M = ptrB->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B(K,M, ctx = ctx);
    
    vcl_B = viennacl::linalg::element_asin(vcl_A);
    
    ptrB->to_host(vcl_B);
}


template <typename T>
void cpp_gpuMatrix_elem_sinh(
    SEXP ptrA_, 
    SEXP ptrB_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    
    const int K = ptrB->nrow();
    const int M = ptrB->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B(K,M, ctx = ctx);
    
    vcl_B = viennacl::linalg::element_sinh(vcl_A);
    
    ptrB->to_host(vcl_B);
}


template <typename T>
void cpp_gpuMatrix_elem_cos(
    SEXP ptrA_, 
    SEXP ptrB_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    
    const int K = ptrB->nrow();
    const int M = ptrB->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B(K,M, ctx = ctx);
    
    vcl_B = viennacl::linalg::element_cos(vcl_A);
    
    ptrB->to_host(vcl_B);
}

template <typename T>
void cpp_gpuMatrix_elem_acos(
    SEXP ptrA_, 
    SEXP ptrB_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    
    const int K = ptrB->nrow();
    const int M = ptrB->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B(K,M, ctx = ctx);
    
    vcl_B = viennacl::linalg::element_acos(vcl_A);
    
    ptrB->to_host(vcl_B);
}


template <typename T>
void cpp_gpuMatrix_elem_cosh(
    SEXP ptrA_, 
    SEXP ptrB_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    
    const int K = ptrB->nrow();
    const int M = ptrB->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B(K,M, ctx = ctx);
    
    vcl_B = viennacl::linalg::element_cosh(vcl_A);
    
    ptrB->to_host(vcl_B);
}


template <typename T>
void cpp_gpuMatrix_elem_tan(
    SEXP ptrA_, 
    SEXP ptrB_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    
    const int K = ptrB->nrow();
    const int M = ptrB->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B(K,M, ctx = ctx);
    
    vcl_B = viennacl::linalg::element_tan(vcl_A);
    
    ptrB->to_host(vcl_B);
}

template <typename T>
void cpp_gpuMatrix_elem_atan(
    SEXP ptrA_, 
    SEXP ptrB_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    
    const int K = ptrB->nrow();
    const int M = ptrB->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B(K,M, ctx = ctx);
    
    vcl_B = viennacl::linalg::element_atan(vcl_A);
    
    ptrB->to_host(vcl_B);
}


template <typename T>
void cpp_gpuMatrix_elem_tanh(
    SEXP ptrA_, 
    SEXP ptrB_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    
    const int K = ptrB->nrow();
    const int M = ptrB->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B(K,M, ctx = ctx);
    
    vcl_B = viennacl::linalg::element_tanh(vcl_A);
    
    ptrB->to_host(vcl_B);
}

template <typename T>
void cpp_gpuMatrix_elem_log(
    SEXP ptrA_, SEXP ptrB_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    
    const int K = ptrB->nrow();
    const int M = ptrB->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B(K,M, ctx = ctx);
    
    vcl_B = viennacl::linalg::element_log(vcl_A);
    
    ptrB->to_host(vcl_B);
}

template <typename T>
void cpp_gpuMatrix_elem_log_base(
    SEXP ptrA_, SEXP ptrB_,
    T base,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    
    const int K = ptrB->nrow();
    const int M = ptrB->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B(K,M, ctx = ctx);
    
    vcl_B = viennacl::linalg::element_log10(vcl_A);
    vcl_B /= log10(base);
    
    ptrB->to_host(vcl_B);
}

template <typename T>
void cpp_gpuMatrix_elem_log10(
    SEXP ptrA_, SEXP ptrB_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    
    const int K = ptrB->nrow();
    const int M = ptrB->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B(K,M, ctx = ctx);
    
    vcl_B = viennacl::linalg::element_log10(vcl_A);
    
    ptrB->to_host(vcl_B);
}

template <typename T>
void cpp_gpuMatrix_elem_exp(
    SEXP ptrA_, SEXP ptrB_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    
    const int K = ptrB->nrow();
    const int M = ptrB->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B(K,M, ctx = ctx);
    
    vcl_B = viennacl::linalg::element_exp(vcl_A);
    
    ptrB->to_host(vcl_B);
}

template <typename T>
void cpp_gpuMatrix_elem_abs(
    SEXP ptrA_, SEXP ptrB_,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    XPtr<dynEigenMat<T> > ptrB(ptrB_);
    
    const int K = ptrB->nrow();
    const int M = ptrB->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<T> vcl_B(K,M, ctx = ctx);
    
    vcl_B = viennacl::linalg::element_fabs(vcl_A);
    
    ptrB->to_host(vcl_B);
}

/*** gpuMatrix Functions ***/

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_prod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_prod<int>(ptrA, ptrB, ptrC, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_prod<float>(ptrA, ptrB, ptrC, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_prod<double>(ptrA, ptrB, ptrC, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_scalar_prod(
    SEXP ptrC,
    SEXP scalar,
    const int type_flag,
    int ctx_id)
{    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_scalar_prod<int>(ptrC, scalar, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_scalar_prod<float>(ptrC, scalar, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_scalar_prod<double>(ptrC, scalar, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_scalar_div(
    SEXP ptrC,
    SEXP B_scalar,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_scalar_div<int>(ptrC, B_scalar, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_scalar_div<float>(ptrC, B_scalar, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_scalar_div<double>(ptrC, B_scalar, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_div(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_div<int>(ptrA, ptrB, ptrC, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_div<float>(ptrA, ptrB, ptrC, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_div<double>(ptrA, ptrB, ptrC, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_pow(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_pow<int>(ptrA, ptrB, ptrC, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_pow<float>(ptrA, ptrB, ptrC, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_pow<double>(ptrA, ptrB, ptrC, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_scalar_pow(
    SEXP ptrA, SEXP scalar, SEXP ptrC,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_scalar_pow<int>(ptrA, scalar, ptrC, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_scalar_pow<float>(ptrA, scalar, ptrC, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_scalar_pow<double>(ptrA, scalar, ptrC, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_sin(
    SEXP ptrA, SEXP ptrB,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_sin<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_sin<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_sin<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_asin(
    SEXP ptrA, SEXP ptrB,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_asin<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_asin<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_asin<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_sinh(
    SEXP ptrA, SEXP ptrB,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_sinh<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_sinh<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_sinh<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_cos(
    SEXP ptrA, SEXP ptrB,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_cos<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_cos<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_cos<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_acos(
    SEXP ptrA, SEXP ptrB,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_acos<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_acos<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_acos<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_cosh(
    SEXP ptrA, SEXP ptrB,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_cosh<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_cosh<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_cosh<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_tan(
    SEXP ptrA, SEXP ptrB,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_tan<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_tan<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_tan<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_atan(
    SEXP ptrA, SEXP ptrB,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_atan<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_atan<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_atan<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_tanh(
    SEXP ptrA, SEXP ptrB,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_tanh<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_tanh<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_tanh<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_log(
    SEXP ptrA, SEXP ptrB,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_log<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_log<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_log<double>(ptrA, ptrB, ctx_id);
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
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_log_base<int>(ptrA, ptrB, as<int>(base), ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_log_base<float>(ptrA, ptrB, as<float>(base), ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_log_base<double>(ptrA, ptrB, as<double>(base), ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_log10(
    SEXP ptrA, SEXP ptrB,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_log10<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_log10<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_log10<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_exp(
    SEXP ptrA, SEXP ptrB,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_exp<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_exp<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_exp<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_abs(
    SEXP ptrA, SEXP ptrB,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_abs<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_abs<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_abs<double>(ptrA, ptrB, ctx_id);
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
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_axpy<int>(alpha, ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_axpy<float>(alpha, ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_axpy<double>(alpha, ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}
    
// [[Rcpp::export]]
void
cpp_gpuMatrix_unary_axpy(
    SEXP ptrA,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_unary_axpy<int>(ptrA, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_unary_axpy<float>(ptrA, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_unary_axpy<double>(ptrA, ctx_id);
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
    SEXP ptrB_)
{
    const T alpha = as<T>(alpha_);
    
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pB(ptrB_);
    
    viennacl::vector_range<viennacl::vector<T> > ptrA  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > ptrB  = pB->data();
    
    ptrB += alpha * (ptrA);
}

template <typename T>
void 
cpp_vclVector_unary_axpy(
    SEXP ptrA_)
{
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    viennacl::vector_range<viennacl::vector<T> > vcl_A  = pA->data();
    
    viennacl::vector<T> vcl_Z = viennacl::zero_vector<T>(vcl_A.size());
    
    vcl_Z -= vcl_A;
    vcl_A = vcl_Z;
}

template <typename T>
T cpp_vclVector_inner_prod(
    SEXP ptrA_, 
    SEXP ptrB_)
{
    T out;
    
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pB(ptrB_);
    
    viennacl::vector_range<viennacl::vector<T> > ptrA  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > ptrB  = pB->data();
    
    out = viennacl::linalg::inner_prod(ptrA, ptrB);
    return out;
}


template <typename T>
void cpp_vclVector_outer_prod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_)
{
    
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pB(ptrB_);
    Rcpp::XPtr<dynVCLMat<T> > pC(ptrC_);
    
    viennacl::vector_range<viennacl::vector<T> > ptrA  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > ptrB  = pB->data();
    viennacl::matrix_range<viennacl::matrix<T> > ptrC  = pC->data();

    ptrC = viennacl::linalg::outer_prod(ptrA, ptrB);
}


template <typename T>
void cpp_vclVector_elem_prod(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_)
{    
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pB(ptrB_);
    Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
    
    viennacl::vector_range<viennacl::vector<T> > ptrA  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > ptrB  = pB->data();
    viennacl::vector_range<viennacl::vector<T> > ptrC  = pC->data();

    ptrC = viennacl::linalg::element_prod(ptrA, ptrB);
}

template <typename T>
void 
cpp_vclVector_scalar_prod(
    SEXP ptrC_, 
    SEXP scalar)
{        
    const T alpha = as<T>(scalar);
    
    Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
    viennacl::vector_range<viennacl::vector<T> > vcl_C  = pC->data();
    
    vcl_C *= alpha;
}

template <typename T>
void cpp_vclVector_elem_div(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_)
{        
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pB(ptrB_);
    Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
    
    viennacl::vector_range<viennacl::vector<T> > ptrA  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > ptrB  = pB->data();
    viennacl::vector_range<viennacl::vector<T> > ptrC  = pC->data();

    ptrC = viennacl::linalg::element_div(ptrA, ptrB);
}

template <typename T>
void 
cpp_vclVector_scalar_div(
    SEXP ptrC_, 
    SEXP scalar)
{        
    const T alpha = as<T>(scalar);
    
    Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
    viennacl::vector_range<viennacl::vector<T> > vcl_C  = pC->data();
    
    vcl_C /= alpha;
}

template <typename T>
void cpp_vclVector_elem_pow(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_)
{    
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pB(ptrB_);
    Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
    
    viennacl::vector_range<viennacl::vector<T> > ptrA  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > ptrB  = pB->data();
    viennacl::vector_range<viennacl::vector<T> > ptrC  = pC->data();

    ptrC = viennacl::linalg::element_pow(ptrA, ptrB);
}

template <typename T>
void cpp_vclVector_scalar_pow(
    SEXP ptrA_, 
    SEXP scalar_, 
    SEXP ptrC_)
{    
    const T scalar = as<T>(scalar_);    
    
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
    
    viennacl::vector_range<viennacl::vector<T> > vcl_A  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > vcl_C  = pC->data();
    
    
    viennacl::vector<T> vcl_B = viennacl::scalar_vector<T>(vcl_A.size(), scalar);
    
    vcl_C = viennacl::linalg::element_pow(vcl_A, vcl_B);
    
}

template <typename T>
void cpp_vclVector_elem_sin(
    SEXP ptrA_, 
    SEXP ptrC_)
{    
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
    
    viennacl::vector_range<viennacl::vector<T> > ptrA  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > ptrC  = pC->data();

    ptrC = viennacl::linalg::element_sin(ptrA);
}


template <typename T>
void cpp_vclVector_elem_asin(
    SEXP ptrA_, 
    SEXP ptrC_)
{    
    
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
    
    viennacl::vector_range<viennacl::vector<T> > ptrA  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > ptrC  = pC->data();

    ptrC = viennacl::linalg::element_asin(ptrA);
}


template <typename T>
void cpp_vclVector_elem_sinh(
    SEXP ptrA_, 
    SEXP ptrC_)
{    

    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
    
    viennacl::vector_range<viennacl::vector<T> > ptrA  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > ptrC  = pC->data();

    ptrC = viennacl::linalg::element_sinh(ptrA);
}


template <typename T>
void cpp_vclVector_elem_cos(
    SEXP ptrA_, 
    SEXP ptrC_)
{    
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
    
    viennacl::vector_range<viennacl::vector<T> > ptrA  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > ptrC  = pC->data();

    ptrC = viennacl::linalg::element_cos(ptrA);
}


template <typename T>
void cpp_vclVector_elem_acos(
    SEXP ptrA_, 
    SEXP ptrC_)
{    
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
    
    viennacl::vector_range<viennacl::vector<T> > ptrA  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > ptrC  = pC->data();

    ptrC = viennacl::linalg::element_acos(ptrA);
}


template <typename T>
void cpp_vclVector_elem_cosh(
    SEXP ptrA_, 
    SEXP ptrC_)
{   
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
    
    viennacl::vector_range<viennacl::vector<T> > ptrA  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > ptrC  = pC->data();

    ptrC = viennacl::linalg::element_cosh(ptrA);
}


template <typename T>
void cpp_vclVector_elem_tan(
    SEXP ptrA_, 
    SEXP ptrC_)
{    
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
    
    viennacl::vector_range<viennacl::vector<T> > ptrA  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > ptrC  = pC->data();

    ptrC = viennacl::linalg::element_tan(ptrA);
}


template <typename T>
void cpp_vclVector_elem_atan(
    SEXP ptrA_, 
    SEXP ptrC_)
{    
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
    
    viennacl::vector_range<viennacl::vector<T> > ptrA  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > ptrC  = pC->data();

    ptrC = viennacl::linalg::element_atan(ptrA);
}


template <typename T>
void cpp_vclVector_elem_tanh(
    SEXP ptrA_, 
    SEXP ptrC_)
{    
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
    
    viennacl::vector_range<viennacl::vector<T> > ptrA  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > ptrC  = pC->data();

    ptrC = viennacl::linalg::element_tanh(ptrA);
}

template <typename T>
void cpp_vclVector_elem_exp(
    SEXP ptrA_, 
    SEXP ptrC_)
{    
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
    
    viennacl::vector_range<viennacl::vector<T> > ptrA  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > ptrC  = pC->data();

    ptrC = viennacl::linalg::element_exp(ptrA);
}


template <typename T>
void cpp_vclVector_elem_log10(
    SEXP ptrA_, 
    SEXP ptrC_)
{ 
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
    
    viennacl::vector_range<viennacl::vector<T> > ptrA  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > ptrC  = pC->data();

    ptrC = viennacl::linalg::element_log10(ptrA);
}


template <typename T>
void cpp_vclVector_elem_log_base(
    SEXP ptrA_, 
    SEXP ptrC_,
    T base)
{
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
    
    viennacl::vector_range<viennacl::vector<T> > ptrA  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > ptrC  = pC->data();

    ptrC = viennacl::linalg::element_log10(ptrA);
    ptrC /= log10(base);
}

template <typename T>
void cpp_vclVector_elem_log(
    SEXP ptrA_, 
    SEXP ptrC_)
{    
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
    
    viennacl::vector_range<viennacl::vector<T> > ptrA  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > ptrC  = pC->data();

    ptrC = viennacl::linalg::element_log(ptrA);
}

template <typename T>
void 
cpp_vclVector_elem_abs(
    SEXP ptrA_, 
    SEXP ptrC_)
{    
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
    
    viennacl::vector_range<viennacl::vector<T> > vcl_A  = pA->data();
    viennacl::vector_range<viennacl::vector<T> > vcl_C  = pC->data();
    
    vcl_C = viennacl::linalg::element_fabs(vcl_A);
}

template <typename T>
T
cpp_vclVector_max(
    SEXP ptrA_)
{    
    T max;
    
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    viennacl::vector_range<viennacl::vector<T> > vcl_A  = pA->data();
    
    max = viennacl::linalg::max(vcl_A);
    
    return max;
}

template <typename T>
T
cpp_vclVector_min(
    SEXP ptrA_)
{    
    T max;
    
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    viennacl::vector_range<viennacl::vector<T> > vcl_A  = pA->data();
    
    max = viennacl::linalg::min(vcl_A);
    
    return max;
}

/*** vclMatrix templates ***/

template <typename T>
void cpp_vclMatrix_axpy(
    SEXP alpha_, 
    SEXP ptrA_, 
    SEXP ptrB_)
{
    const T alpha = as<T>(alpha_);
    
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B  = ptrB->data();
    
    B += alpha * (A);
}

template <typename T>
void 
cpp_vclMatrix_unary_axpy(
    SEXP ptrA_,
    int ctx_id)
{
   
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);    
    viennacl::matrix_range<viennacl::matrix<T> > vcl_A  = ptrA->data();
    viennacl::context ctx(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
    
    
    viennacl::matrix<T> vcl_Z = viennacl::zero_matrix<T>(vcl_A.size1(),vcl_A.size2(), ctx);
    
    vcl_Z -= vcl_A;
    vcl_A = vcl_Z;
}


template <typename T>
void cpp_vclMatrix_elem_prod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_)
{    
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    Rcpp::XPtr<dynVCLMat<T> > ptrC(ptrC_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B  = ptrB->data();
    viennacl::matrix_range<viennacl::matrix<T> > C  = ptrC->data();

    C = viennacl::linalg::element_prod(A, B);
}

template <typename T>
void 
cpp_vclMatrix_scalar_prod(
    SEXP ptrC_, 
    SEXP scalar)
{        
    const T alpha = as<T>(scalar);

    Rcpp::XPtr<dynVCLMat<T> > ptrC(ptrC_);
    viennacl::matrix_range<viennacl::matrix<T> > vcl_C  = ptrC->data();
    
    vcl_C *= alpha;
}

template <typename T>
void cpp_vclMatrix_elem_div(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_)
{
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    Rcpp::XPtr<dynVCLMat<T> > ptrC(ptrC_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B  = ptrB->data();
    viennacl::matrix_range<viennacl::matrix<T> > C  = ptrC->data();

    C = viennacl::linalg::element_div(A, B);
}

template <typename T>
void 
cpp_vclMatrix_scalar_div(
    SEXP ptrC_, 
    SEXP scalar)
{        
    const T alpha = as<T>(scalar);

    Rcpp::XPtr<dynVCLMat<T> > ptrC(ptrC_);
    viennacl::matrix_range<viennacl::matrix<T> > vcl_C  = ptrC->data();
    
    vcl_C /= alpha;
}

template <typename T>
void cpp_vclMatrix_elem_pow(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_)
{
    
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    Rcpp::XPtr<dynVCLMat<T> > ptrC(ptrC_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B  = ptrB->data();
    viennacl::matrix_range<viennacl::matrix<T> > C  = ptrC->data();

    C = viennacl::linalg::element_pow(A, B);
}

template <typename T>
void 
cpp_vclMatrix_scalar_pow(
    SEXP ptrA_, 
    SEXP scalar_, 
    SEXP ptrC_,
    int ctx_id)
{    
    
    const T scalar = as<T>(scalar_);    
    
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrC(ptrC_);
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    viennacl::matrix_range<viennacl::matrix<T> > vcl_A  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > vcl_C  = ptrC->data();
    
    viennacl::matrix<T> vcl_B = viennacl::scalar_matrix<T>(vcl_A.size1(),vcl_A.size2(),scalar, ctx);
    
    vcl_C = viennacl::linalg::element_pow(vcl_A, vcl_B);
}

template <typename T>
void cpp_vclMatrix_elem_sin(
    SEXP ptrA_, 
    SEXP ptrB_)
{
    
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B  = ptrB->data();

    B = viennacl::linalg::element_sin(A);
}

template <typename T>
void cpp_vclMatrix_elem_asin(
    SEXP ptrA_, 
    SEXP ptrB_)
{

    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B  = ptrB->data();

    B = viennacl::linalg::element_asin(A);
}

template <typename T>
void cpp_vclMatrix_elem_sinh(
    SEXP ptrA_, 
    SEXP ptrB_)
{
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B  = ptrB->data();

    B = viennacl::linalg::element_sinh(A);
}

template <typename T>
void cpp_vclMatrix_elem_cos(
    SEXP ptrA_, 
    SEXP ptrB_)
{    
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B  = ptrB->data();

    B = viennacl::linalg::element_cos(A);
}

template <typename T>
void cpp_vclMatrix_elem_acos(
    SEXP ptrA_, 
    SEXP ptrB_)
{

    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B  = ptrB->data();

    B = viennacl::linalg::element_acos(A);
}

template <typename T>
void cpp_vclMatrix_elem_cosh(
    SEXP ptrA_, 
    SEXP ptrB_)
{
    
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B  = ptrB->data();

    B = viennacl::linalg::element_cosh(A);
}

template <typename T>
void cpp_vclMatrix_elem_tan(
    SEXP ptrA_, 
    SEXP ptrB_)
{
    
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B  = ptrB->data();

    B = viennacl::linalg::element_tan(A);
}

template <typename T>
void cpp_vclMatrix_elem_atan(
    SEXP ptrA_, 
    SEXP ptrB_)
{
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B  = ptrB->data();

    B = viennacl::linalg::element_atan(A);
}

template <typename T>
void cpp_vclMatrix_elem_tanh(
    SEXP ptrA_, 
    SEXP ptrB_)
{    
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B  = ptrB->data();

    B = viennacl::linalg::element_tanh(A);
}

template <typename T>
void cpp_vclMatrix_elem_log(
    SEXP ptrA_, 
    SEXP ptrB_)
{    

    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B  = ptrB->data();

    B = viennacl::linalg::element_log(A);
}


template <typename T>
void cpp_vclMatrix_elem_log10(
    SEXP ptrA_, 
    SEXP ptrB_)
{    
    
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B  = ptrB->data();

    B = viennacl::linalg::element_log10(A);
}


template <typename T>
void cpp_vclMatrix_elem_log_base(
    SEXP ptrA_, 
    SEXP ptrB_,
    const float base)
{

    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B  = ptrB->data();

    B = viennacl::linalg::element_log10(A);
    B /= log10(base);
}

template <typename T>
void cpp_vclMatrix_elem_exp(
    SEXP ptrA_, 
    SEXP ptrB_)
{        
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B  = ptrB->data();

    B = viennacl::linalg::element_exp(A);
}

template <typename T>
void cpp_vclMatrix_elem_abs(
    SEXP ptrA_, 
    SEXP ptrB_)
{        
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    
    viennacl::matrix_range<viennacl::matrix<T> > vcl_A  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > vcl_B  = ptrB->data();
    
    vcl_B = viennacl::linalg::element_fabs(vcl_A);
}

template <typename T>
T
cpp_vclMatrix_max(
    SEXP ptrA_)
{    
    
    T max_out;
    
    Rcpp::XPtr<dynVCLMat<T> > pA(ptrA_);
    viennacl::matrix_range<viennacl::matrix<T> > vcl_A  = pA->data();
    
    // iterate over columns
    Rcpp::NumericVector max_vec(vcl_A.size2());
    
    for(unsigned int i=0; i<vcl_A.size2(); i++){
        max_vec[i] = viennacl::linalg::max(viennacl::column(vcl_A, i));
    }
    
    max_out = max(max_vec);
    
    return max_out;
}

template <typename T>
T
cpp_vclMatrix_min(
    SEXP ptrA_)
{    
    T min_out;
    
    Rcpp::XPtr<dynVCLMat<T> > pA(ptrA_);
    viennacl::matrix_range<viennacl::matrix<T> > vcl_A  = pA->data();
    
    // iterate over columns
    Rcpp::NumericVector min_vec(vcl_A.size2());
    
    for(unsigned int i=0; i<vcl_A.size2(); i++){
        min_vec[i] = viennacl::linalg::min(viennacl::column(vcl_A, i));
    }
    
    min_out = min(min_vec);
    
    return min_out;
}

/*** vclMatrix Functions ***/

// [[Rcpp::export]]
void
cpp_vclMatrix_axpy(
    SEXP alpha,
    SEXP ptrA, SEXP ptrB,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclMatrix_axpy<int>(alpha, ptrA, ptrB);
            return;
        case 6:
            cpp_vclMatrix_axpy<float>(alpha, ptrA, ptrB);
            return;
        case 8:
            cpp_vclMatrix_axpy<double>(alpha, ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclMatrix_unary_axpy(
    SEXP ptrA,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclMatrix_unary_axpy<int>(ptrA, ctx_id);
            return;
        case 6:
            cpp_vclMatrix_unary_axpy<float>(ptrA, ctx_id);
            return;
        case 8:
            cpp_vclMatrix_unary_axpy<double>(ptrA, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

//[[Rcpp::export]]
void cpp_vclMatrix_elem_prod(
    SEXP ptrA, 
    SEXP ptrB,
    SEXP ptrC,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_prod<int>(ptrA, ptrB, ptrC);
            return;
        case 6:
            cpp_vclMatrix_elem_prod<float>(ptrA, ptrB, ptrC);
            return;
        case 8:
            cpp_vclMatrix_elem_prod<double>(ptrA, ptrB, ptrC);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclMatrix_scalar_prod(
    SEXP ptrC,
    SEXP B_scalar,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclMatrix_scalar_prod<int>(ptrC, B_scalar);
            return;
        case 6:
            cpp_vclMatrix_scalar_prod<float>(ptrC, B_scalar);
            return;
        case 8:
            cpp_vclMatrix_scalar_prod<double>(ptrC, B_scalar);
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
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_div<int>(ptrA, ptrB, ptrC);
            return;
        case 6:
            cpp_vclMatrix_elem_div<float>(ptrA, ptrB, ptrC);
            return;
        case 8:
            cpp_vclMatrix_elem_div<double>(ptrA, ptrB, ptrC);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclMatrix_scalar_div(
    SEXP ptrC,
    SEXP B_scalar,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclMatrix_scalar_div<int>(ptrC, B_scalar);
            return;
        case 6:
            cpp_vclMatrix_scalar_div<float>(ptrC, B_scalar);
            return;
        case 8:
            cpp_vclMatrix_scalar_div<double>(ptrC, B_scalar);
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
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_pow<int>(ptrA, ptrB, ptrC);
            return;
        case 6:
            cpp_vclMatrix_elem_pow<float>(ptrA, ptrB, ptrC);
            return;
        case 8:
            cpp_vclMatrix_elem_pow<double>(ptrA, ptrB, ptrC);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclMatrix_scalar_pow(
    SEXP ptrA, 
    SEXP scalar, 
    SEXP ptrC,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclMatrix_scalar_pow<int>(ptrA, scalar, ptrC, ctx_id);
            return;
        case 6:
            cpp_vclMatrix_scalar_pow<float>(ptrA, scalar, ptrC, ctx_id);
            return;
        case 8:
            cpp_vclMatrix_scalar_pow<double>(ptrA, scalar, ptrC, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

//[[Rcpp::export]]
void cpp_vclMatrix_elem_sin(
    SEXP ptrA, 
    SEXP ptrB,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_sin<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclMatrix_elem_sin<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclMatrix_elem_sin<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

//[[Rcpp::export]]
void cpp_vclMatrix_elem_asin(
    SEXP ptrA, 
    SEXP ptrB,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_asin<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclMatrix_elem_asin<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclMatrix_elem_asin<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

//[[Rcpp::export]]
void cpp_vclMatrix_elem_sinh(
    SEXP ptrA, 
    SEXP ptrB,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_sinh<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclMatrix_elem_sinh<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclMatrix_elem_sinh<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


//[[Rcpp::export]]
void cpp_vclMatrix_elem_cos(
    SEXP ptrA, 
    SEXP ptrB,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_cos<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclMatrix_elem_cos<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclMatrix_elem_cos<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

//[[Rcpp::export]]
void cpp_vclMatrix_elem_acos(
    SEXP ptrA, 
    SEXP ptrB,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_acos<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclMatrix_elem_acos<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclMatrix_elem_acos<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


//[[Rcpp::export]]
void cpp_vclMatrix_elem_cosh(
    SEXP ptrA, 
    SEXP ptrB,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_cosh<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclMatrix_elem_cosh<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclMatrix_elem_cosh<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


//[[Rcpp::export]]
void cpp_vclMatrix_elem_tan(
    SEXP ptrA, 
    SEXP ptrB,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_tan<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclMatrix_elem_tan<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclMatrix_elem_tan<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

//[[Rcpp::export]]
void cpp_vclMatrix_elem_atan(
    SEXP ptrA, 
    SEXP ptrB,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_atan<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclMatrix_elem_atan<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclMatrix_elem_atan<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

//[[Rcpp::export]]
void cpp_vclMatrix_elem_tanh(
    SEXP ptrA, 
    SEXP ptrB,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_tanh<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclMatrix_elem_tanh<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclMatrix_elem_tanh<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

//[[Rcpp::export]]
void cpp_vclMatrix_elem_log(
    SEXP ptrA, 
    SEXP ptrB,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_log<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclMatrix_elem_log<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclMatrix_elem_log<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


//[[Rcpp::export]]
void cpp_vclMatrix_elem_log10(
    SEXP ptrA, 
    SEXP ptrB,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_log10<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclMatrix_elem_log10<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclMatrix_elem_log10<double>(ptrA, ptrB);
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
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_log_base<int>(ptrA, ptrB, as<int>(base));
            return;
        case 6:
            cpp_vclMatrix_elem_log_base<float>(ptrA, ptrB, as<float>(base));
            return;
        case 8:
            cpp_vclMatrix_elem_log_base<double>(ptrA, ptrB, as<double>(base));
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

//[[Rcpp::export]]
void cpp_vclMatrix_elem_exp(
    SEXP ptrA, 
    SEXP ptrB,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_exp<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclMatrix_elem_exp<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclMatrix_elem_exp<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

//[[Rcpp::export]]
void cpp_vclMatrix_elem_abs(
    SEXP ptrA, 
    SEXP ptrB,
    const int type_flag)
{
   switch(type_flag) {
        case 4:
            cpp_vclMatrix_elem_abs<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclMatrix_elem_abs<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclMatrix_elem_abs<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
SEXP
cpp_vclMatrix_max(
    SEXP ptrA,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            return wrap(cpp_vclMatrix_max<int>(ptrA));
        case 6:
            return wrap(cpp_vclMatrix_max<float>(ptrA));
        case 8:
            return wrap(cpp_vclMatrix_max<double>(ptrA));
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
SEXP
cpp_vclMatrix_min(
    SEXP ptrA,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            return wrap(cpp_vclMatrix_min<int>(ptrA));
        case 6:
            return wrap(cpp_vclMatrix_min<float>(ptrA));
        case 8:
            return wrap(cpp_vclMatrix_min<double>(ptrA));
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
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_axpy<int>(alpha, ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuVector_axpy<float>(alpha, ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuVector_axpy<double>(alpha, ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_unary_axpy(
    SEXP ptrA,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_unary_axpy<int>(ptrA, ctx_id);
            return;
        case 6:
            cpp_gpuVector_unary_axpy<float>(ptrA, ctx_id);
            return;
        case 8:
            cpp_gpuVector_unary_axpy<double>(ptrA, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}


// [[Rcpp::export]]
SEXP
cpp_gpuVector_inner_prod(
    SEXP ptrA, SEXP ptrB,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            return wrap(cpp_gpuVector_inner_prod<int>(ptrA, ptrB, ctx_id));
        case 6:
            return wrap(cpp_gpuVector_inner_prod<float>(ptrA, ptrB, ctx_id));
        case 8:
            return wrap(cpp_gpuVector_inner_prod<double>(ptrA, ptrB, ctx_id));
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_outer_prod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_outer_prod<int>(ptrA, ptrB, ptrC, ctx_id);
            return;
        case 6:
            cpp_gpuVector_outer_prod<float>(ptrA, ptrB, ptrC, ctx_id);
            return;
        case 8:
            cpp_gpuVector_outer_prod<double>(ptrA, ptrB, ptrC, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_gpuVector_elem_prod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_prod<int>(ptrA, ptrB, ptrC, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_prod<float>(ptrA, ptrB, ptrC, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_prod<double>(ptrA, ptrB, ptrC, ctx_id);
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
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_scalar_prod<int>(ptrC, scalar, ctx_id);
            return;
        case 6:
            cpp_gpuVector_scalar_prod<float>(ptrC, scalar, ctx_id);
            return;
        case 8:
            cpp_gpuVector_scalar_prod<double>(ptrC, scalar, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_div(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_div<int>(ptrA, ptrB, ptrC, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_div<float>(ptrA, ptrB, ptrC, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_div<double>(ptrA, ptrB, ptrC, ctx_id);
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
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_scalar_div<int>(ptrC, scalar, order, ctx_id);
            return;
        case 6:
            cpp_gpuVector_scalar_div<float>(ptrC, scalar, order, ctx_id);
            return;
        case 8:
            cpp_gpuVector_scalar_div<double>(ptrC, scalar, order, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_pow(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_pow<int>(ptrA, ptrB, ptrC, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_pow<float>(ptrA, ptrB, ptrC, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_pow<double>(ptrA, ptrB, ptrC, ctx_id);
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
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_scalar_pow<int>(ptrA, scalar, ptrC, order, ctx_id);
            return;
        case 6:
            cpp_gpuVector_scalar_pow<float>(ptrA, scalar, ptrC, order, ctx_id);
            return;
        case 8:
            cpp_gpuVector_scalar_pow<double>(ptrA, scalar, ptrC, order, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_sin(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_sin<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_sin<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_sin<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_asin(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_asin<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_asin<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_asin<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_sinh(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_sinh<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_sinh<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_sinh<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_cos(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_cos<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_cos<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_cos<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_acos(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_acos<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_acos<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_acos<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_cosh(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_cosh<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_cosh<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_cosh<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_gpuVector_elem_tan(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_tan<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_tan<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_tan<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_atan(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_atan<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_atan<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_atan<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_tanh(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_tanh<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_tanh<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_tanh<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_log10(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_log10<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_log10<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_log10<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_log(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_log<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_log<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_log<double>(ptrA, ptrB, ctx_id);
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
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_log_base<int>(ptrA, ptrB, as<int>(base), ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_log_base<float>(ptrA, ptrB, as<float>(base), ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_log_base<double>(ptrA, ptrB, as<double>(base), ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_exp(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_exp<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_exp<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_exp<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_abs(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_abs<int>(ptrA, ptrB, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_abs<float>(ptrA, ptrB, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_abs<double>(ptrA, ptrB, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
SEXP
cpp_gpuVector_max(
    SEXP ptrA,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            return wrap(cpp_gpuVector_max<int>(ptrA, ctx_id));
        case 6:
            return wrap(cpp_gpuVector_max<float>(ptrA, ctx_id));
        case 8:
            return wrap(cpp_gpuVector_max<double>(ptrA, ctx_id));
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
SEXP
cpp_gpuVector_min(
    SEXP ptrA,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            return wrap(cpp_gpuVector_min<int>(ptrA, ctx_id));
        case 6:
            return wrap(cpp_gpuVector_min<float>(ptrA, ctx_id));
        case 8:
            return wrap(cpp_gpuVector_min<double>(ptrA, ctx_id));
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
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_axpy<int>(alpha, ptrA, ptrB);
            return;
        case 6:
            cpp_vclVector_axpy<float>(alpha, ptrA, ptrB);
            return;
        case 8:
            cpp_vclVector_axpy<double>(alpha, ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_unary_axpy(
    SEXP ptrA,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_unary_axpy<int>(ptrA);
            return;
        case 6:
            cpp_vclVector_unary_axpy<float>(ptrA);
            return;
        case 8:
            cpp_vclVector_unary_axpy<double>(ptrA);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
SEXP
cpp_vclVector_inner_prod(
    SEXP ptrA, SEXP ptrB,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            return wrap(cpp_vclVector_inner_prod<int>(ptrA, ptrB));
        case 6:
            return wrap(cpp_vclVector_inner_prod<float>(ptrA, ptrB));
        case 8:
            return wrap(cpp_vclVector_inner_prod<double>(ptrA, ptrB));
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_outer_prod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_outer_prod<int>(ptrA, ptrB, ptrC);
            return;
        case 6:
            cpp_vclVector_outer_prod<float>(ptrA, ptrB, ptrC);
            return;
        case 8:
            cpp_vclVector_outer_prod<double>(ptrA, ptrB, ptrC);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_elem_prod(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_prod<int>(ptrA, ptrB, ptrC);
            return;
        case 6:
            cpp_vclVector_elem_prod<float>(ptrA, ptrB, ptrC);
            return;
        case 8:
            cpp_vclVector_elem_prod<double>(ptrA, ptrB, ptrC);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_scalar_prod(
    SEXP ptrC,
    SEXP scalar,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_scalar_prod<int>(ptrC, scalar);
            return;
        case 6:
            cpp_vclVector_scalar_prod<float>(ptrC, scalar);
            return;
        case 8:
            cpp_vclVector_scalar_prod<double>(ptrC, scalar);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_elem_div(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_div<int>(ptrA, ptrB, ptrC);
            return;
        case 6:
            cpp_vclVector_elem_div<float>(ptrA, ptrB, ptrC);
            return;
        case 8:
            cpp_vclVector_elem_div<double>(ptrA, ptrB, ptrC);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

    
// [[Rcpp::export]]
void
cpp_vclVector_scalar_div(
    SEXP ptrC, 
    SEXP scalar, 
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_scalar_div<int>(ptrC, scalar);
            return;
        case 6:
            cpp_vclVector_scalar_div<float>(ptrC, scalar);
            return;
        case 8:
            cpp_vclVector_scalar_div<double>(ptrC, scalar);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_elem_pow(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_pow<int>(ptrA, ptrB, ptrC);
            return;
        case 6:
            cpp_vclVector_elem_pow<float>(ptrA, ptrB, ptrC);
            return;
        case 8:
            cpp_vclVector_elem_pow<double>(ptrA, ptrB, ptrC);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

    
// [[Rcpp::export]]
void
cpp_vclVector_scalar_pow(
    SEXP ptrA, 
    SEXP scalar, 
    SEXP ptrC,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_scalar_pow<int>(ptrA, scalar, ptrC);
            return;
        case 6:
            cpp_vclVector_scalar_pow<float>(ptrA, scalar, ptrC);
            return;
        case 8:
            cpp_vclVector_scalar_pow<double>(ptrA, scalar, ptrC);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_elem_sin(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_sin<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclVector_elem_sin<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclVector_elem_sin<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_elem_asin(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_asin<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclVector_elem_asin<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclVector_elem_asin<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclVector_elem_sinh(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_sinh<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclVector_elem_sinh<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclVector_elem_sinh<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclVector_elem_cos(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_cos<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclVector_elem_cos<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclVector_elem_cos<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_elem_acos(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_acos<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclVector_elem_acos<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclVector_elem_acos<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclVector_elem_cosh(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_cosh<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclVector_elem_cosh<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclVector_elem_cosh<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclVector_elem_tan(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_tan<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclVector_elem_tan<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclVector_elem_tan<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_elem_atan(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_atan<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclVector_elem_atan<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclVector_elem_atan<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclVector_elem_tanh(
    SEXP ptrA, SEXP ptrB,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_tanh<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclVector_elem_tanh<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclVector_elem_tanh<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclVector_elem_log(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_log<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclVector_elem_log<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclVector_elem_log<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_elem_log10(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_log10<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclVector_elem_log10<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclVector_elem_log10<double>(ptrA, ptrB);
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
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_log_base<int>(ptrA, ptrB, as<int>(R_base));
            return;
        case 6:
            cpp_vclVector_elem_log_base<float>(ptrA, ptrB, as<float>(R_base));
            return;
        case 8:
            cpp_vclVector_elem_log_base<double>(ptrA, ptrB, as<double>(R_base));
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclVector_elem_exp(
    SEXP ptrA, SEXP ptrB, 
    const int type_flag)
{
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_exp<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclVector_elem_exp<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclVector_elem_exp<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_elem_abs(
    SEXP ptrA, SEXP ptrC, 
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclVector_elem_abs<int>(ptrA, ptrC);
            return;
        case 6:
            cpp_vclVector_elem_abs<float>(ptrA, ptrC);
            return;
        case 8:
            cpp_vclVector_elem_abs<double>(ptrA, ptrC);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
SEXP
cpp_vclVector_max(
    SEXP ptrA,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            return wrap(cpp_vclVector_max<int>(ptrA));
        case 6:
            return wrap(cpp_vclVector_max<float>(ptrA));
        case 8:
            return wrap(cpp_vclVector_max<double>(ptrA));
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
SEXP
cpp_vclVector_min(
    SEXP ptrA,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            return wrap(cpp_vclVector_min<int>(ptrA));
        case 6:
            return wrap(cpp_vclVector_min<float>(ptrA));
        case 8:
            return wrap(cpp_vclVector_min<double>(ptrA));
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

