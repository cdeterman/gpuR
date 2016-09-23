
#ifndef GET_VCL_PTR_HPP
#define GET_VCL_PTR_HPP

#include "gpuR/dynVCLMat.hpp"
#include "gpuR/dynEigenMat.hpp"

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

#endif
