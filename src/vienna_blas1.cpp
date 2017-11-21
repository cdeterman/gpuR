
#include "gpuR/windows_check.hpp"

// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/utils.hpp"
#include "gpuR/getVCLptr.hpp"
// #include "gpuR/dynEigenMat.hpp"
#include "gpuR/dynEigenVec.hpp"
// #include "gpuR/dynVCLMat.hpp"
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
    SEXP A_, 
    const bool AisVCL,
    SEXP B_,
    const bool BisVCL,
    const int order,
    const int ctx_id)
{
    
    const T alpha = as<T>(alpha_);
    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(A_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_B = getVCLVecptr<T>(B_, BisVCL, ctx_id);
    
    if(order == 0){
        *vcl_B += alpha * *vcl_A; 
    }else{   
        *vcl_B = alpha * *vcl_B + *vcl_A;
    }
    
    if(!BisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrB(B_);
        
        // copy device data back to CPU
        ptrB->to_host(*vcl_B);
        ptrB->release_device();
    }
}

template <typename T>
void 
cpp_gpuVector_unary_axpy(
    SEXP ptrA_,
    const bool AisVCL,
    int ctx_id)
{
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));

    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    
    viennacl::vector_base<T> vcl_Z = viennacl::vector_base<T>(vcl_A->size(), ctx = ctx);
    viennacl::linalg::vector_assign(vcl_Z, (T)(0));
    
    vcl_Z -= *vcl_A;
    
    if(!AisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrA(ptrA_);
        
        // copy device data back to CPU
        ptrA->to_host(vcl_Z);
        ptrA->release_device();
    }else{
        *vcl_A = vcl_Z;
    }
    // viennacl::copy(vcl_Z, Am);
    // viennacl::fast_copy(vcl_Z.begin(), vcl_Z.end(), &(Am[0]));
}


template <typename T>
T cpp_gpuVector_inner_prod(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrB_,
    const bool BisVCL,
    int ctx_id)
{   
    T C;    
    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_B = getVCLVecptr<T>(ptrB_, BisVCL, ctx_id);
    
    C = viennacl::linalg::inner_prod(*vcl_A, *vcl_B);
    
    return C;
}

template <typename T>
void cpp_gpuVector_outer_prod(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrB_, 
    const bool BisVCL,
    SEXP ptrC_,
    const bool CisVCL,
    int ctx_id)
{   
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_B = getVCLVecptr<T>(ptrB_, BisVCL, ctx_id);
    
    std::shared_ptr<viennacl::matrix<T> > vcl_C = getVCLptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::outer_prod(*vcl_A, *vcl_B);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenMat<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
}

template <typename T>
void cpp_gpuVector_elem_prod(
        SEXP ptrA_, 
        const bool AisVCL,
        SEXP ptrB_, 
        const bool BisVCL,
        SEXP ptrC_,
        const bool CisVCL,
        int ctx_id)
{    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_B = getVCLVecptr<T>(ptrB_, BisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_prod(*vcl_A, *vcl_B);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
}

template <typename T>
void 
cpp_gpuVector_scalar_prod(
    SEXP ptrC_, 
    const bool CisVCL,
    SEXP scalar,
    int ctx_id)
{        
    const T alpha = as<T>(scalar);
    
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C *= alpha;
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
}

template <typename T>
void cpp_gpuVector_elem_div(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrB_, 
    const bool BisVCL,
    SEXP ptrC_,
    const bool CisVCL,
    int ctx_id)
{    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_B = getVCLVecptr<T>(ptrB_, BisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_div(*vcl_A, *vcl_B);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
}

template <typename T>
void 
cpp_gpuVector_scalar_div(
    SEXP ptrC_, 
    const bool CisVCL,
    SEXP scalar, 
    const int order,
    int ctx_id)
{        
    const T alpha = as<T>(scalar);
    
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    if(order == 0){
        *vcl_C /= alpha;
        
        if(!CisVCL){
            Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
            
            // copy device data back to CPU
            ptrC->to_host(*vcl_C);
            ptrC->release_device();
        }
    }else{
        viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
        
        viennacl::vector_base<T> vcl_scalar = viennacl::vector_base<T>(vcl_C->size(), ctx = ctx);
        viennacl::linalg::vector_assign(vcl_scalar, alpha);
        
        *vcl_C = viennacl::linalg::element_div(vcl_scalar, *vcl_C);
        
        if(!CisVCL){
            Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
            
            // copy device data back to CPU
            ptrC->to_host(*vcl_C);
            ptrC->release_device();
        }
    }
}

template <typename T>
void cpp_gpuVector_elem_pow(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrB_, 
    const bool BisVCL,
    SEXP ptrC_,
    const bool CisVCL,
    int ctx_id)
{   
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_B = getVCLVecptr<T>(ptrB_, BisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_pow(*vcl_A, *vcl_B);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
}

template <typename T>
void cpp_gpuVector_scalar_pow(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP scalar_, 
    SEXP ptrC_,
    const bool CisVCL,
    const int order,
    int ctx_id)
{    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    const T scalar = as<T>(scalar_);
    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    viennacl::vector_base<T> vcl_B = viennacl::vector_base<T>(vcl_A->size(), ctx = ctx);
    viennacl::linalg::vector_assign(vcl_B, scalar);
    
    if(order == 0){
        *vcl_C = viennacl::linalg::element_pow(*vcl_A, vcl_B);
    }else{
        *vcl_C = viennacl::linalg::element_pow(vcl_B, *vcl_A);
    }
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
}

template <typename T>
void 
cpp_gpuVector_sqrt(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrC_,
    const bool CisVCL,
    int ctx_id)
{    
    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_sqrt(*vcl_A);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
}

template <typename T>
void 
cpp_gpuVector_elem_sin(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrC_,
    const bool CisVCL,
    int ctx_id)
{    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_sin(*vcl_A);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
}

template <typename T>
void 
cpp_gpuVector_elem_asin(
    SEXP ptrA_, 
    const bool AisVCL, 
    SEXP ptrC_,
    const bool CisVCL,
    int ctx_id)
{    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_asin(*vcl_A);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
}

template <typename T>
void 
cpp_gpuVector_elem_sinh(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrC_,
    const bool CisVCL,
    int ctx_id)
{    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_sinh(*vcl_A);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
}

template <typename T>
void 
cpp_gpuVector_elem_cos(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrC_,
    const bool CisVCL,
    int ctx_id)
{    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_cos(*vcl_A);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
}

template <typename T>
void 
cpp_gpuVector_elem_acos(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrC_,
    const bool CisVCL,
    int ctx_id)
{    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_acos(*vcl_A);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
}

template <typename T>
void 
cpp_gpuVector_elem_cosh(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrC_,
    const bool CisVCL,
    int ctx_id)
{    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_cosh(*vcl_A);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
}

template <typename T>
void 
cpp_gpuVector_elem_tan(
    SEXP ptrA_,
    const bool AisVCL, 
    SEXP ptrC_,
    const bool CisVCL,
    int ctx_id)
{    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_tan(*vcl_A);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
}

template <typename T>
void 
cpp_gpuVector_elem_atan(
    SEXP ptrA_,
    const bool AisVCL, 
    SEXP ptrC_,
    const bool CisVCL,
    int ctx_id)
{    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_atan(*vcl_A);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
}

template <typename T>
void 
cpp_gpuVector_elem_tanh(
    SEXP ptrA_, 
    const bool AisVCL, 
    SEXP ptrC_,
    const bool CisVCL,
    int ctx_id)
{    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_tanh(*vcl_A);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
}

template <typename T>
void 
cpp_gpuVector_elem_exp(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrC_,
    const bool CisVCL,
    int ctx_id)
{    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_exp(*vcl_A);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
}

template <typename T>
void 
cpp_gpuVector_elem_log10(
    SEXP ptrA_, 
    const bool AisVCL, 
    SEXP ptrC_,
    const bool CisVCL,
    int ctx_id)
{    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_log10(*vcl_A);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
}

template <typename T>
void 
cpp_gpuVector_elem_log(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrC_,
    const bool CisVCL,
    int ctx_id)
{    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_log(*vcl_A);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
}

template <typename T>
void 
cpp_gpuVector_elem_log_base(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrC_,
    const bool CisVCL,
    T base,
    int ctx_id)
{    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_log10(*vcl_A);
    *vcl_C /= log10(base);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
}

template <typename T>
void 
cpp_gpuVector_elem_abs(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrC_,
    const bool CisVCL,
    int ctx_id)
{    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_C = getVCLVecptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_fabs(*vcl_A);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device();
    }
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
    
    viennacl::vector_base<T> vcl_A(M, ctx = ctx);
    
    // viennacl::copy(Am, vcl_A); 
    viennacl::fast_copy(Am.data(), Am.data() + Am.size(), vcl_A.begin());
    
    max = viennacl::linalg::min(vcl_A);
    
    return max;
}

/*** gpuMatrix Templates ***/

template <typename T>
void 
cpp_gpuMatrix_axpy(
    SEXP alpha_, 
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrB_,
    const bool BisVCL,
    const int ctx_id)
{
    const T alpha = as<T>(alpha_);
    
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
    
    *vcl_B += alpha * *vcl_A;
    
    if(!BisVCL){
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        
        // copy device data back to CPU
        ptrB->to_host(*vcl_B);
        ptrB->release_device();
    }
}

template <typename T>
void 
cpp_gpuMatrix_unary_axpy(
    SEXP ptrA_)
{
    XPtr<dynEigenMat<T> > ptrA(ptrA_);
    
    viennacl::context ctx(viennacl::ocl::get_context(ptrA->getContext()));
    
    const int M = ptrA->nrow();
    const int K = ptrA->ncol();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data();
    viennacl::matrix<T> vcl_Z = viennacl::zero_matrix<T>(M,K, ctx);
    
    vcl_Z -= vcl_A;

    ptrA->to_host(vcl_Z);
}

template <typename T>
void cpp_gpuMatrix_scalar_axpy(
        SEXP alpha_, 
        SEXP scalar_, 
        SEXP ptrC_,
        const bool CisVCL,
        const int order,
        int max_local_size,
        SEXP sourceCode_,
        const int ctx_id)
{
    const T alpha = as<T>(alpha_);
    const T scalar = as<T>(scalar_);
    
    std::string my_kernel = as<std::string>(sourceCode_);
    viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    std::shared_ptr<viennacl::matrix<T> > vcl_C = getVCLptr<T>(ptrC_, CisVCL, ctx_id);
    
    int M = vcl_C->size1();
    // int N = vcl_B.size1();
    int P = vcl_C->size2();
    int M_internal = vcl_C->internal_size1();
    int P_internal = vcl_C->internal_size2();
    
    // add kernel to program
    viennacl::ocl::program & my_prog = ctx.add_program(my_kernel, "my_kernel");
    
    // get compiled kernel function
    viennacl::ocl::kernel & my_kernel_mul = my_prog.get_kernel("scalar_axpy");
    
    cl_device_type type_check = ctx.current_device().type();
    
    if(type_check & CL_DEVICE_TYPE_CPU){
        max_local_size = 1;
    }else{
        cl_device_id raw_device = ctx.current_device().id();
        cl_kernel raw_kernel = ctx.get_kernel("my_kernel", "scalar_axpy").handle().get();
        size_t preferred_work_group_size_multiple;
        
        cl_int err = clGetKernelWorkGroupInfo(raw_kernel, raw_device, 
                                              CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                              sizeof(size_t), &preferred_work_group_size_multiple, NULL);
        
        if(err != CL_SUCCESS){
            Rcpp::stop("Acquiring kernel work group info failed");
        }
        
        max_local_size = roundDown(max_local_size, preferred_work_group_size_multiple);
    }
    
    // set global work sizes
    my_kernel_mul.global_work_size(0, M_internal);
    my_kernel_mul.global_work_size(1, P_internal);
    
    // set local work sizes
    my_kernel_mul.local_work_size(0, max_local_size);
    my_kernel_mul.local_work_size(1, max_local_size);
    
    // execute kernel
    viennacl::ocl::enqueue(my_kernel_mul(*vcl_C, scalar, alpha, order, M, P, P_internal));
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenMat<T> > ptrC(ptrC_);
        ptrC->to_host(*vcl_C);
        ptrC->release_device();    
    }
}

template <typename T>
void 
cpp_gpuMatrix_elem_prod(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrB_, 
    const bool BisVCL,
    SEXP ptrC_,
    const bool CisVCL,
    const int ctx_id)
{   
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
    std::shared_ptr<viennacl::matrix<T> > vcl_C = getVCLptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_prod(*vcl_A, *vcl_B);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenMat<T> > ptrC(ptrC_);
        ptrC->to_host(*vcl_C);
        ptrC->release_device();    
    }
}

template <typename T>
void 
cpp_gpuMatrix_scalar_prod(
    SEXP ptrC_, 
    const bool CisVCL,
    SEXP scalar,
    const int ctx_id)
{        
    const T alpha = as<T>(scalar);
    
    std::shared_ptr<viennacl::matrix<T> > vcl_C = getVCLptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C *= alpha;
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenMat<T> > ptrC(ptrC_);
        ptrC->to_host(*vcl_C);
        ptrC->release_device();    
    }
}

template <typename T>
void 
cpp_gpuMatrix_elem_div(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrB_, 
    const bool BisVCL,
    SEXP ptrC_,
    const bool CisVCL,
    const int ctx_id)
{   
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
    std::shared_ptr<viennacl::matrix<T> > vcl_C = getVCLptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_div(*vcl_A, *vcl_B);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenMat<T> > ptrC(ptrC_);
        ptrC->to_host(*vcl_C);
        ptrC->release_device();    
    }
}

template <typename T>
void 
cpp_gpuMatrix_scalar_div(
    SEXP ptrC_,
    const bool CisVCL,
    SEXP B_scalar,
    const int ctx_id)
{    
    T B = Rcpp::as<T>(B_scalar);
    
    std::shared_ptr<viennacl::matrix<T> > vcl_C = getVCLptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C /= B;
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenMat<T> > ptrC(ptrC_);
        ptrC->to_host(*vcl_C);
        ptrC->release_device();    
    }
}

template <typename T>
void cpp_gpuMatrix_scalar_div_2(
        SEXP scalar,
        SEXP ptrC_,
        const bool CisVCL,
        int max_local_size,
        SEXP sourceCode_,
        const int ctx_id)
{
    // declarations
    const T alpha = as<T>(scalar);
    
    std::string my_kernel = as<std::string>(sourceCode_);
    viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    std::shared_ptr<viennacl::matrix<T> > vcl_C = getVCLptr<T>(ptrC_, CisVCL, ctx_id);
    
    int M = vcl_C->size1();
    // int N = vcl_B.size1();
    int P = vcl_C->size2();
    int M_internal = vcl_C->internal_size1();
    int P_internal = vcl_C->internal_size2();
    
    // add kernel to program
    viennacl::ocl::program & my_prog = ctx.add_program(my_kernel, "my_kernel");
    
    // get compiled kernel function
    viennacl::ocl::kernel & my_kernel_mul = my_prog.get_kernel("ScalarElemDiv");
    
    cl_device_type type_check = ctx.current_device().type();
    
    if(type_check & CL_DEVICE_TYPE_CPU){
        max_local_size = 1;
    }else{
        cl_device_id raw_device = ctx.current_device().id();
        cl_kernel raw_kernel = ctx.get_kernel("my_kernel", "ScalarElemDiv").handle().get();
        size_t preferred_work_group_size_multiple;
        
        cl_int err = clGetKernelWorkGroupInfo(raw_kernel, raw_device, 
                                              CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                              sizeof(size_t), &preferred_work_group_size_multiple, NULL);
        
        if(err != CL_SUCCESS){
            Rcpp::stop("Acquiring kernel work group info failed");
        }
        
        max_local_size = roundDown(max_local_size, preferred_work_group_size_multiple);
    }
    
    // set global work sizes
    my_kernel_mul.global_work_size(0, M_internal);
    my_kernel_mul.global_work_size(1, P_internal);
    
    // set local work sizes
    my_kernel_mul.local_work_size(0, max_local_size);
    my_kernel_mul.local_work_size(1, max_local_size);
    
    // execute kernel
    viennacl::ocl::enqueue(my_kernel_mul(*vcl_C, alpha, M, P, P_internal));
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenMat<T> > ptrC(ptrC_);
        ptrC->to_host(*vcl_C);
        ptrC->release_device();    
    }
}

template <typename T>
void 
cpp_gpuMatrix_elem_pow(
    SEXP ptrA_,
    const bool AisVCL,
    SEXP ptrB_, 
    const bool BisVCL,
    SEXP ptrC_,
    const bool CisVCL,
    const int ctx_id)
{    
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
    std::shared_ptr<viennacl::matrix<T> > vcl_C = getVCLptr<T>(ptrC_, CisVCL, ctx_id);
    
    *vcl_C = viennacl::linalg::element_pow(*vcl_A, *vcl_B);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenMat<T> > ptrC(ptrC_);
        ptrC->to_host(*vcl_C);
        ptrC->release_device();    
    }
}

template <typename T>
void 
cpp_gpuMatrix_scalar_pow(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP scalar_, 
    SEXP ptrC_,
    const bool CisVCL,
    const int ctx_id)
{   
    const T scalar = as<T>(scalar_);    
    
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::matrix<T> > vcl_C = getVCLptr<T>(ptrC_, CisVCL, ctx_id);
    
    // XPtr<dynEigenMat<T> > ptrA(ptrA_);
    // XPtr<dynEigenMat<T> > ptrC(ptrC_);
    
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    const int K = vcl_C->size1();
    const int M = vcl_C->size2();
    
    // viennacl::matrix<T> vcl_A = ptrA->device_data();
    // viennacl::matrix<T> vcl_C(K,M, ctx = ctx);
    
    viennacl::matrix<T> vcl_B = viennacl::scalar_matrix<T>(K,M,scalar, ctx = ctx);
    
    *vcl_C = viennacl::linalg::element_pow(*vcl_A, vcl_B);
    
    if(!CisVCL){
        Rcpp::XPtr<dynEigenMat<T> > ptrC(ptrC_);
        ptrC->to_host(*vcl_C);
        ptrC->release_device();    
    }
}

template <typename T>
void cpp_gpuMatrix_sqrt(
        SEXP ptrA_, 
        const bool AisVCL,
        SEXP ptrB_,
        const bool BisVCL,
        const int ctx_id)
{    
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    
    if(BisVCL){
        std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
        *vcl_B = viennacl::linalg::element_sqrt(*vcl_A);        
    }else{
        viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
        viennacl::matrix<T> vcl_B(vcl_A->size1(),vcl_A->size2(), ctx = ctx);
        vcl_B = viennacl::linalg::element_sqrt(*vcl_A);
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        ptrB->to_host(vcl_B);
        ptrB->release_device();    
    }
}

template <typename T>
void cpp_gpuMatrix_elem_sin(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrB_,
    const bool BisVCL,
    const int ctx_id)
{    
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    
    if(BisVCL){
        std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
        *vcl_B = viennacl::linalg::element_sin(*vcl_A);        
    }else{
        viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
        viennacl::matrix<T> vcl_B(vcl_A->size1(),vcl_A->size2(), ctx = ctx);
        vcl_B = viennacl::linalg::element_sin(*vcl_A);
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        ptrB->to_host(vcl_B);
        ptrB->release_device();    
    }
}

template <typename T>
void cpp_gpuMatrix_elem_asin(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrB_,
    const bool BisVCL,
    const int ctx_id)
{    
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    
    if(BisVCL){
        std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
        *vcl_B = viennacl::linalg::element_asin(*vcl_A);        
    }else{
        viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
        viennacl::matrix<T> vcl_B(vcl_A->size1(),vcl_A->size2(), ctx = ctx);
        vcl_B = viennacl::linalg::element_asin(*vcl_A);
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        ptrB->to_host(vcl_B);
        ptrB->release_device();    
    }
}


template <typename T>
void cpp_gpuMatrix_elem_sinh(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrB_,
    const bool BisVCL,
    const int ctx_id)
{    
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    
    if(BisVCL){
        std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
        *vcl_B = viennacl::linalg::element_sinh(*vcl_A);        
    }else{
        viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
        viennacl::matrix<T> vcl_B(vcl_A->size1(),vcl_A->size2(), ctx = ctx);
        vcl_B = viennacl::linalg::element_sinh(*vcl_A);
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        ptrB->to_host(vcl_B);
        ptrB->release_device();    
    }
}


template <typename T>
void cpp_gpuMatrix_elem_cos(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrB_,
    const bool BisVCL,
    const int ctx_id)
{    
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    
    if(BisVCL){
        std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
        *vcl_B = viennacl::linalg::element_cos(*vcl_A);        
    }else{
        viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
        viennacl::matrix<T> vcl_B(vcl_A->size1(),vcl_A->size2(), ctx = ctx);
        vcl_B = viennacl::linalg::element_cos(*vcl_A);
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        ptrB->to_host(vcl_B);
        ptrB->release_device();    
    }
}

template <typename T>
void cpp_gpuMatrix_elem_acos(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrB_,
    const bool BisVCL,
    const int ctx_id)
{    
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    
    if(BisVCL){
        std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
        *vcl_B = viennacl::linalg::element_acos(*vcl_A);        
    }else{
        viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
        viennacl::matrix<T> vcl_B(vcl_A->size1(),vcl_A->size2(), ctx = ctx);
        vcl_B = viennacl::linalg::element_acos(*vcl_A);
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        ptrB->to_host(vcl_B);
        ptrB->release_device();    
    }
}


template <typename T>
void cpp_gpuMatrix_elem_cosh(
    SEXP ptrA_,
    const bool AisVCL,
    SEXP ptrB_,
    const bool BisVCL,
    const int ctx_id)
{   
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    
    if(BisVCL){
        std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
        *vcl_B = viennacl::linalg::element_cosh(*vcl_A);        
    }else{
        viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
        viennacl::matrix<T> vcl_B(vcl_A->size1(),vcl_A->size2(), ctx = ctx);
        vcl_B = viennacl::linalg::element_cosh(*vcl_A);
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        ptrB->to_host(vcl_B);
        ptrB->release_device();    
    }
}


template <typename T>
void cpp_gpuMatrix_elem_tan(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrB_,
    const bool BisVCL,
    const int ctx_id)
{    
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    
    if(BisVCL){
        std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
        *vcl_B = viennacl::linalg::element_tan(*vcl_A);        
    }else{
        viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
        viennacl::matrix<T> vcl_B(vcl_A->size1(),vcl_A->size2(), ctx = ctx);
        vcl_B = viennacl::linalg::element_tan(*vcl_A);
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        ptrB->to_host(vcl_B);
        ptrB->release_device();    
    }
}

template <typename T>
void cpp_gpuMatrix_elem_atan(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrB_,
    const bool BisVCL,
    const int ctx_id)
{    
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    
    if(BisVCL){
        std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
        *vcl_B = viennacl::linalg::element_atan(*vcl_A);        
    }else{
        viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
        viennacl::matrix<T> vcl_B(vcl_A->size1(),vcl_A->size2(), ctx = ctx);
        vcl_B = viennacl::linalg::element_atan(*vcl_A);
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        ptrB->to_host(vcl_B);
        ptrB->release_device();    
    }
}


template <typename T>
void cpp_gpuMatrix_elem_tanh(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrB_,
    const bool BisVCL,
    const int ctx_id)
{    
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    
    if(BisVCL){
        std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
        *vcl_B = viennacl::linalg::element_tanh(*vcl_A);        
    }else{
        viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
        viennacl::matrix<T> vcl_B(vcl_A->size1(),vcl_A->size2(), ctx = ctx);
        vcl_B = viennacl::linalg::element_tanh(*vcl_A);
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        ptrB->to_host(vcl_B);
        ptrB->release_device();    
    }
}

template <typename T>
void cpp_gpuMatrix_elem_log(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrB_,
    const bool BisVCL,
    const int ctx_id)
{    
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    
    if(BisVCL){
        std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
        *vcl_B = viennacl::linalg::element_log(*vcl_A);        
    }else{
        viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
        viennacl::matrix<T> vcl_B(vcl_A->size1(),vcl_A->size2(), ctx = ctx);
        vcl_B = viennacl::linalg::element_log(*vcl_A);
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        ptrB->to_host(vcl_B);
        ptrB->release_device();    
    }
}

template <typename T>
void cpp_gpuMatrix_elem_log_base(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrB_,
    const bool BisVCL,
    T base,
    const int ctx_id
)
{    
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    
    if(BisVCL){
        std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
        *vcl_B = viennacl::linalg::element_log10(*vcl_A);   
        *vcl_B /= log10(base);     
    }else{
        viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
        viennacl::matrix<T> vcl_B(vcl_A->size1(),vcl_A->size2(), ctx = ctx);
        vcl_B = viennacl::linalg::element_log10(*vcl_A);
        vcl_B /= log10(base);
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        ptrB->to_host(vcl_B);
        ptrB->release_device();    
    }
}

template <typename T>
void cpp_gpuMatrix_elem_log10(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrB_,
    const bool BisVCL,
    const int ctx_id)
{    
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    
    if(BisVCL){
        std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
        *vcl_B = viennacl::linalg::element_log10(*vcl_A);        
    }else{
        viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
        viennacl::matrix<T> vcl_B(vcl_A->size1(),vcl_A->size2(), ctx = ctx);
        vcl_B = viennacl::linalg::element_log10(*vcl_A);
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        ptrB->to_host(vcl_B);
        ptrB->release_device();    
    }
}

template <typename T>
void cpp_gpuMatrix_elem_exp(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrB_,
    const bool BisVCL,
    const int ctx_id)
{    
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    
    if(BisVCL){
        std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
        *vcl_B = viennacl::linalg::element_exp(*vcl_A);        
    }else{
        viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
        viennacl::matrix<T> vcl_B(vcl_A->size1(),vcl_A->size2(), ctx = ctx);
        vcl_B = viennacl::linalg::element_exp(*vcl_A);
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        ptrB->to_host(vcl_B);
        ptrB->release_device();    
    }
}

template <typename T>
void cpp_gpuMatrix_elem_abs(
    SEXP ptrA_, 
    const bool AisVCL,
    SEXP ptrB_,
    const bool BisVCL,
    const int ctx_id)
{    
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    
    if(BisVCL){
        std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
        *vcl_B = viennacl::linalg::element_fabs(*vcl_A);        
    }else{
        viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
        viennacl::matrix<T> vcl_B(vcl_A->size1(),vcl_A->size2(), ctx = ctx);
        vcl_B = viennacl::linalg::element_fabs(*vcl_A);
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        ptrB->to_host(vcl_B);
        ptrB->release_device();    
    }
}

/*** gpuMatrix Functions ***/

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_prod(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB, 
    const bool BisVCL, 
    SEXP ptrC,
    const bool CisVCL,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_prod<int>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_prod<float>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_prod<double>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_scalar_prod(
    SEXP ptrC,
    const bool CisVCL,
    SEXP scalar,
    const int type_flag,
    const int ctx_id)
{    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_scalar_prod<int>(ptrC, CisVCL, scalar, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_scalar_prod<float>(ptrC, CisVCL, scalar, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_scalar_prod<double>(ptrC, CisVCL, scalar, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_scalar_div(
    SEXP ptrC,
    const bool CisVCL,
    SEXP B_scalar,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_scalar_div<int>(ptrC, CisVCL, B_scalar, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_scalar_div<float>(ptrC, CisVCL, B_scalar, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_scalar_div<double>(ptrC, CisVCL, B_scalar, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_scalar_div_2(
    SEXP ptrC,
    const bool CisVCL,
    SEXP scalar,
    int max_local_size,
    SEXP sourceCode_,
    const int ctx_id,
    const int type_flag)
{
    
    switch(type_flag) {
    case 4:
        cpp_gpuMatrix_scalar_div_2<int>(scalar, ptrC, CisVCL, max_local_size, sourceCode_, ctx_id);
        return;
    case 6:
        cpp_gpuMatrix_scalar_div_2<float>(scalar, ptrC, CisVCL, max_local_size, sourceCode_, ctx_id);
        return;
    case 8:
        cpp_gpuMatrix_scalar_div_2<double>(scalar, ptrC, CisVCL, max_local_size, sourceCode_, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_div(
    SEXP ptrA,
    const bool AisVCL, 
    SEXP ptrB, 
    const bool BisVCL,
    SEXP ptrC,
    const bool CisVCL,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_div<int>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_div<float>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_div<double>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_pow(
    SEXP ptrA,
    const bool AisVCL,
    SEXP ptrB, 
    const bool BisVCL,
    SEXP ptrC,
    const bool CisVCL,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_pow<int>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_pow<float>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_pow<double>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_scalar_pow(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP scalar, 
    SEXP ptrC,
    const bool CisVCL,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_scalar_pow<int>(ptrA, AisVCL, scalar, ptrC, CisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_scalar_pow<float>(ptrA, AisVCL, scalar, ptrC, CisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_scalar_pow<double>(ptrA, AisVCL, scalar, ptrC, CisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_sqrt(
    SEXP ptrA, 
    const bool AisVCL, 
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_sqrt<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_sqrt<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_sqrt<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_sin(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_sin<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_sin<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_sin<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_asin(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_asin<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_asin<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_asin<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_sinh(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_sinh<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_sinh<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_sinh<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_cos(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_cos<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_cos<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_cos<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_acos(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_acos<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_acos<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_acos<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_cosh(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_cosh<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_cosh<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_cosh<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_tan(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_tan<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_tan<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_tan<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_atan(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_atan<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_atan<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_atan<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_tanh(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_tanh<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_tanh<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_tanh<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_log(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_log<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_log<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_log<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_log_base(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    SEXP base,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_log_base<int>(ptrA, AisVCL, ptrB, BisVCL, as<int>(base), ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_log_base<float>(ptrA, AisVCL, ptrB, BisVCL, as<int>(base), ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_log_base<double>(ptrA, AisVCL, ptrB, BisVCL, as<int>(base), ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_log10(
    SEXP ptrA,
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_log10<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_log10<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_log10<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_exp(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_exp<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_exp<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_exp<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_elem_abs(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_elem_abs<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_elem_abs<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_elem_abs<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_axpy(
    SEXP alpha,
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_axpy<int>(alpha, ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuMatrix_axpy<float>(alpha, ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_axpy<double>(alpha, ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}
    
// [[Rcpp::export]]
void
cpp_gpuMatrix_unary_axpy(
    SEXP ptrA,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_unary_axpy<int>(ptrA);
            return;
        case 6:
            cpp_gpuMatrix_unary_axpy<float>(ptrA);
            return;
        case 8:
            cpp_gpuMatrix_unary_axpy<double>(ptrA);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_scalar_axpy(
    SEXP alpha,
    SEXP scalar, 
    SEXP ptrB,
    const bool BisVCL,
    const int order,
    int max_local_size,
    SEXP sourceCode,
    const int ctx_id,
    const int type_flag)
{
    
    switch(type_flag) {
    case 4:
        cpp_gpuMatrix_scalar_axpy<int>(alpha, scalar, ptrB, BisVCL, order, max_local_size, sourceCode, ctx_id);
        return;
    case 6:
        cpp_gpuMatrix_scalar_axpy<float>(alpha, scalar, ptrB, BisVCL, order, max_local_size, sourceCode, ctx_id);
        return;
    case 8:
        cpp_gpuMatrix_scalar_axpy<double>(alpha, scalar, ptrB, BisVCL, order, max_local_size, sourceCode, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for gpuMatrix object!");
    }
}


/*** vclVector Templates ***/


// template <typename T>
// void 
// cpp_vclVector_scalar_div(
//     SEXP ptrC_, 
//     SEXP scalar)
// {        
//     const T alpha = as<T>(scalar);
//     
//     Rcpp::XPtr<dynVCLVec<T> > pC(ptrC_);
//     viennacl::vector_range<viennacl::vector_base<T> > vcl_C  = pC->data();
//     
//     vcl_C /= alpha;
// }

template <typename T>
T
cpp_vclVector_elem_max_abs(
    SEXP ptrA_)
{    
    T max;
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    
    viennacl::vector_range<viennacl::vector_base<T> > vcl_A  = pA->data();
    
    // max = viennacl::linalg::max(viennacl::linalg::element_fabs(vcl_A));
    
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am(vcl_A.size());
    
    viennacl::fast_copy(vcl_A.begin(), vcl_A.end(), &(Am[0]));
    // viennacl::fast_copy(block.data(), block.data() + block.size(), shptr.get()->begin());
    
    max = Am.cwiseAbs().maxCoeff();
    
    return max;
}


// probably want to have some sort of switch for when to leave on the
// device and when to copy to host
template <typename T>
T
cpp_vclVector_max(
    SEXP ptrA_)
{    
    // T max;
    
    // Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    // viennacl::vector_range<viennacl::vector_base<T> > vcl_A  = pA->data();
    // max = viennacl::linalg::max(vcl_A);
    
    Rcpp::XPtr<dynVCLVec<T> > ptrA(ptrA_);
    viennacl::vector_range<viennacl::vector_base<T> > tempA = ptrA->data();
    
    viennacl::vector_base<T> pA = static_cast<viennacl::vector_base<T> >(tempA);
    int M = pA.size();
    
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am(M);
    
    // viennacl::copy(pA, Am); 
    viennacl::fast_copy(pA.begin(), pA.end(), &(Am[0]));
    
    return Am.maxCoeff();
}

template <typename T>
T
cpp_vclVector_min(
    SEXP ptrA_)
{    
    T max;
    
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    viennacl::vector_range<viennacl::vector_base<T> > vcl_A  = pA->data();
    
    max = viennacl::linalg::min(vcl_A);
    
    return max;
}

/*** vclMatrix templates ***/

template <typename T>
void 
cpp_vclMatrix_unary_axpy(
    SEXP ptrA_,
    int ctx_id)
{
   
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);    
    viennacl::matrix_range<viennacl::matrix<T> > vcl_A  = ptrA->data();
    viennacl::context ctx(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
    
    
    // viennacl::matrix<T> vcl_Z = viennacl::zero_matrix<T>(vcl_A.size1(),vcl_A.size2(), ctx);
    // 
    // vcl_Z -= vcl_A;
    // vcl_A = vcl_Z;
    
    vcl_A = (T)(-1) * vcl_A;
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
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    const int order,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_axpy<int>(alpha, ptrA, AisVCL, ptrB, BisVCL, order, ctx_id);
            return;
        case 6:
            cpp_gpuVector_axpy<float>(alpha, ptrA, AisVCL, ptrB, BisVCL, order, ctx_id);
            return;
        case 8:
            cpp_gpuVector_axpy<double>(alpha, ptrA, AisVCL, ptrB, BisVCL, order, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_unary_axpy(
    SEXP ptrA,
    const bool AisVCL,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_unary_axpy<int>(ptrA, AisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuVector_unary_axpy<float>(ptrA, AisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuVector_unary_axpy<double>(ptrA, AisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}


// [[Rcpp::export]]
SEXP
cpp_gpuVector_inner_prod(
    SEXP ptrA,
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            return wrap(cpp_gpuVector_inner_prod<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id));
        case 6:
            return wrap(cpp_gpuVector_inner_prod<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id));
        case 8:
            return wrap(cpp_gpuVector_inner_prod<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id));
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_outer_prod(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB, 
    const bool BisVCL, 
    SEXP ptrC,
    const bool CisVCL,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_outer_prod<int>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuVector_outer_prod<float>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuVector_outer_prod<double>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_gpuVector_elem_prod(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB, 
    const bool BisVCL,
    SEXP ptrC,
    const bool CisVCL,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_prod<int>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_prod<float>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_prod<double>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}
    
// [[Rcpp::export]]
void
cpp_gpuVector_scalar_prod(
    SEXP ptrC,
    const bool CisVCL,
    SEXP scalar,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_scalar_prod<int>(ptrC, CisVCL, scalar, ctx_id);
            return;
        case 6:
            cpp_gpuVector_scalar_prod<float>(ptrC, CisVCL, scalar, ctx_id);
            return;
        case 8:
            cpp_gpuVector_scalar_prod<double>(ptrC, CisVCL, scalar, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_div(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB, 
    const bool BisVCL, 
    SEXP ptrC,
    const bool CisVCL, 
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_div<int>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_div<float>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_div<double>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_scalar_div(
    SEXP ptrC,
    const bool CisVCL,
    SEXP scalar,
    const int order,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_scalar_div<int>(ptrC, CisVCL, scalar, order, ctx_id);
            return;
        case 6:
            cpp_gpuVector_scalar_div<float>(ptrC, CisVCL, scalar, order, ctx_id);
            return;
        case 8:
            cpp_gpuVector_scalar_div<double>(ptrC, CisVCL, scalar, order, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_pow(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB, 
    const bool BisVCL, 
    SEXP ptrC,
    const bool CisVCL, 
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_pow<int>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_pow<float>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_pow<double>(ptrA, AisVCL, ptrB, BisVCL, ptrC, CisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_scalar_pow(
    SEXP ptrA, 
    const bool AisVCL, 
    SEXP scalar, 
    SEXP ptrC,
    const bool CisVCL,
    const int order,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_scalar_pow<int>(ptrA, AisVCL, scalar, ptrC, CisVCL, order, ctx_id);
            return;
        case 6:
            cpp_gpuVector_scalar_pow<float>(ptrA, AisVCL, scalar, ptrC, CisVCL, order, ctx_id);
            return;
        case 8:
            cpp_gpuVector_scalar_pow<double>(ptrA, AisVCL, scalar, ptrC, CisVCL, order, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_sqrt(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB, 
    const bool BisVCL,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
    case 4:
        cpp_gpuVector_sqrt<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
        return;
    case 6:
        cpp_gpuVector_sqrt<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
        return;
    case 8:
        cpp_gpuVector_sqrt<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_sin(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_sin<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_sin<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_sin<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_asin(
    SEXP ptrA, 
    const bool AisVCL, 
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_asin<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_asin<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_asin<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_sinh(
    SEXP ptrA, 
    const bool AisVCL, 
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_sinh<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_sinh<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_sinh<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_cos(
    SEXP ptrA, 
    const bool AisVCL, 
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_cos<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_cos<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_cos<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_acos(
    SEXP ptrA, 
    const bool AisVCL, 
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_acos<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_acos<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_acos<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_cosh(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_cosh<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_cosh<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_cosh<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}


// [[Rcpp::export]]
void
cpp_gpuVector_elem_tan(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_tan<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_tan<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_tan<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_atan(
    SEXP ptrA, 
    const bool AisVCL, 
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_atan<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_atan<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_atan<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_tanh(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_tanh<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_tanh<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_tanh<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_log10(
    SEXP ptrA, 
    const bool AisVCL, 
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_log10<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_log10<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_log10<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_log(
    SEXP ptrA, 
    const bool AisVCL, 
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_log<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_log<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_log<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_log_base(
    SEXP ptrA, 
    const bool AisVCL,
    SEXP ptrB,
    const bool BisVCL,
    SEXP base,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_log_base<int>(ptrA, AisVCL, ptrB, BisVCL, as<int>(base), ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_log_base<float>(ptrA, AisVCL, ptrB, BisVCL, as<float>(base), ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_log_base<double>(ptrA, AisVCL, ptrB, BisVCL, as<double>(base), ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_exp(
    SEXP ptrA, 
    const bool AisVCL, 
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_exp<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_exp<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_exp<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuVector_elem_abs(
    SEXP ptrA, 
    const bool AisVCL, 
    SEXP ptrB,
    const bool BisVCL,
    const int type_flag,
    int ctx_id)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuVector_elem_abs<int>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 6:
            cpp_gpuVector_elem_abs<float>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuVector_elem_abs<double>(ptrA, AisVCL, ptrB, BisVCL, ctx_id);
            return;
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
cpp_vclVector_elem_max_abs(
    SEXP ptrA,
    const int type_flag)
{
    
    switch(type_flag) {
    case 4:
        return wrap(cpp_vclVector_elem_max_abs<int>(ptrA));
    case 6:
        return wrap(cpp_vclVector_elem_max_abs<float>(ptrA));
    case 8:
        return wrap(cpp_vclVector_elem_max_abs<double>(ptrA));
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

