
#include "gpuR/windows_check.hpp"


#include "gpuR/getVCLptr.hpp"

#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/qr.hpp"

using namespace Rcpp;

template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, std::vector<T> >::type
#else
    std::vector<T>
#endif 
cpp_gpuR_qr(
    SEXP ptrA_,
    const bool isVCL,
    const int ctx_id)
{
    
    // viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    // Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    
    // viennacl::matrix_range<viennacl::matrix<T> > A = ptrA->data();
    // viennacl::matrix<T> vcl_A = ptrA->matrix();
    std::shared_ptr<viennacl::matrix<T> > ptrA = getVCLptr<T>(ptrA_, isVCL, ctx_id);
    
    // dereference to create copy
    // likely will add option to allow inplace
    // viennacl::matrix vcl_A = *ptrA;
    
    //computes the QR factorization
    std::vector<T> betas = viennacl::linalg::inplace_qr(*ptrA);
    
    if(!isVCL){
        // move back to host
        Rcpp::XPtr<dynEigenMat<T> > hostA(ptrA_);
        
        // copy device data back to CPU
        hostA->to_host();
        hostA->release_device(); 
    }
    
    return betas;
}

template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else
    void
#endif 
cpp_recover_qr(
    SEXP ptrQR_,
    const bool QRisVCL,
    SEXP ptrQ_,
    const bool QisVCL,
    SEXP ptrR_,
    const bool RisVCL,
    SEXP betas_,
    const int ctx_id)
{

    // viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    std::shared_ptr<viennacl::matrix_range<viennacl::matrix<T> > > vcl_QR = getVCLBlockptr<T>(ptrQR_, QRisVCL, ctx_id);
    std::shared_ptr<viennacl::matrix_range<viennacl::matrix<T> > > vcl_Q = getVCLBlockptr<T>(ptrQ_, QisVCL, ctx_id);
    std::shared_ptr<viennacl::matrix_range<viennacl::matrix<T> > > vcl_R = getVCLBlockptr<T>(ptrR_, RisVCL, ctx_id);
    
    std::vector<T> betas = as<std::vector<T> >(betas_);

    viennacl::linalg::recoverQ(*vcl_QR, betas, *vcl_Q, *vcl_R);
    
    if(!QisVCL){
        // move back to host
        Rcpp::XPtr<dynEigenMat<T> > ptrQ(ptrQ_);
        
        // copy device data back to CPU
        ptrQ->to_host();
        ptrQ->release_device(); 
    }
    
    if(!RisVCL){
        // move back to host
        Rcpp::XPtr<dynEigenMat<T> > ptrR(ptrR_);
        
        // copy device data back to CPU
        ptrR->to_host();
        ptrR->release_device(); 
    }
    
}


// [[Rcpp::export]]
SEXP
cpp_gpuR_qr(
    SEXP ptrA,
    const bool isVCL,
    const int type_flag,
    const int ctx_id)
{

    switch(type_flag) {
#ifndef BACKEND_CUDA
    case 4:
        return wrap(cpp_gpuR_qr<int>(ptrA, isVCL, ctx_id));
        // return wrap(betas);
#endif
    case 6:
        return wrap(cpp_gpuR_qr<float>(ptrA, isVCL, ctx_id));
        // return wrap(betas);
    case 8:
        return wrap(cpp_gpuR_qr<double>(ptrA, isVCL, ctx_id));
        // return wrap(betas);
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_recover_qr(
    SEXP ptrQR,
    const bool QRisVCL,
    SEXP ptrQ,
    const bool QisVCL,
    SEXP ptrR,
    const bool RisVCL,
    SEXP betas,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
#ifndef BACKEND_CUDA
    case 4:
        cpp_recover_qr<int>(ptrQR, QRisVCL, ptrQ, QisVCL, ptrR, RisVCL, betas, ctx_id);
        return;
#endif
    case 6:
        cpp_recover_qr<float>(ptrQR, QRisVCL, ptrQ, QisVCL, ptrR, RisVCL, betas, ctx_id);
        return;
    case 8:
        cpp_recover_qr<double>(ptrQR, QRisVCL, ptrQ, QisVCL, ptrR, RisVCL, betas, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

