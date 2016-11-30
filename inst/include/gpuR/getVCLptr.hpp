
#ifndef GET_VCL_PTR_HPP
#define GET_VCL_PTR_HPP

#include "gpuR/dynVCLMat.hpp"
#include "gpuR/dynVCLVec.hpp"
#include "gpuR/dynEigenMat.hpp"
#include "gpuR/dynEigenVec.hpp"

template<typename T>
viennacl::matrix<T>*
getVCLptr(
    SEXP ptr_,
    const bool isVCL,
    const int ctx_id
){
    viennacl::matrix<T>* vclptr;
    
    if(isVCL){
        Rcpp::XPtr<dynVCLMat<T> > ptr(ptr_);
        vclptr = ptr->getPtr();
    }else{
        Rcpp::XPtr<dynEigenMat<T> > ptr(ptr_);
        ptr->to_device(ctx_id);
        vclptr = ptr->getDevicePtr();
    }
    
    return vclptr;
}

template<typename T>
viennacl::vector_base<T>*
getVCLVecptr(
    SEXP ptr_,
    const bool isVCL,
    const int ctx_id
){
    viennacl::vector_base<T>* vclptr;
    
    if(isVCL){
        Rcpp::XPtr<dynVCLVec<T> > ptr(ptr_);
        vclptr = ptr->getPtr();
    }else{
        Rcpp::XPtr<dynEigenVec<T> > ptr(ptr_);
        
        ptr->to_device(ctx_id);
        vclptr = ptr->getDevicePtr();
    }
    
    return vclptr;
}

#endif
