
#ifndef GET_VCL_PTR_HPP
#define GET_VCL_PTR_HPP

#include "gpuR/dynVCLMat.hpp"
#include "gpuR/dynVCLVec.hpp"
#include "gpuR/dynEigenMat.hpp"
#include "gpuR/dynEigenVec.hpp"

#include <type_traits>

template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, std::shared_ptr<viennacl::matrix<T> > >::type
#else 
std::shared_ptr<viennacl::matrix<T> >    
#endif
getVCLptr(
    SEXP ptr_,
    const bool isVCL,
    const int ctx_id
){
    std::shared_ptr<viennacl::matrix<T> > vclptr;
    
    if(isVCL){
        Rcpp::XPtr<dynVCLMat<T> > ptr(ptr_);
        vclptr = ptr->sharedPtr();
    }else{
        Rcpp::XPtr<dynEigenMat<T> > ptr(ptr_);
        ptr->to_device(ctx_id);
        vclptr = ptr->getDevicePtr();
    }
    
    return vclptr;
}

template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, std::shared_ptr<viennacl::matrix_range<viennacl::matrix<T> > > >::type
#else
std::shared_ptr<viennacl::matrix_range<viennacl::matrix<T> > >
#endif
getVCLBlockptr(
    SEXP ptr_,
    const bool isVCL,
    const int ctx_id
){
    std::shared_ptr<viennacl::matrix_range<viennacl::matrix<T> > > vclptr;
    
    if(isVCL){
        Rcpp::XPtr<dynVCLMat<T> > ptr(ptr_);
        vclptr = ptr->sharedBlockPtr();
    }else{
        Rcpp::XPtr<dynEigenMat<T> > ptr(ptr_);
        ptr->to_device(ctx_id);
        vclptr = ptr->getDeviceBlockPtr();
    }
    
    return vclptr;
}

template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, std::shared_ptr<viennacl::vector_base<T> > >::type
#else
std::shared_ptr<viennacl::vector_base<T> >
#endif
getVCLVecptr(
    SEXP ptr_,
    const bool isVCL,
    const int ctx_id
){
    std::shared_ptr<viennacl::vector_base<T> > vclptr;
    
    if(isVCL){
        Rcpp::XPtr<dynVCLVec<T> > ptr(ptr_);
        vclptr = ptr->sharedPtr();
    }else{
        Rcpp::XPtr<dynEigenVec<T> > ptr(ptr_);
        
        ptr->to_device(ctx_id);
        vclptr = ptr->getDevicePtr();
    }
    
    return vclptr;
}

#endif
