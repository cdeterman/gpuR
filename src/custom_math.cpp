#include "gpuR/windows_check.hpp"

#include <RcppEigen.h>

#include "viennacl/ocl/backend.hpp"

#include "gpuR/utils.hpp"
#include "gpuR/getVCLptr.hpp"

// using namespace cl;
using namespace Rcpp;


template<typename T>
void
cpp_vclMatrix_sign(
    SEXP ptrA_,
    const bool AisVCL,
    SEXP ptrB_, 
    const bool BisVCL,
    SEXP sourceCode_,
    int max_local_size,
    const int ctx_id)
{
    std::string my_kernel = as<std::string>(sourceCode_);
    
    viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
    
    unsigned int M = vcl_B->size1();
    // // int N = vcl_B.size1();
    unsigned int P = vcl_B->size2();
    unsigned int M_internal = vcl_B->internal_size1();
    unsigned int P_internal = vcl_B->internal_size2();
    
    // add kernel to program
    viennacl::ocl::program & my_prog = ctx.add_program(my_kernel, "my_kernel");
    
    // get compiled kernel function
    viennacl::ocl::kernel & mat_sign = my_prog.get_kernel("MatSign");
    
    cl_device_type type_check = ctx.current_device().type();
    
    if(type_check & CL_DEVICE_TYPE_CPU){
        max_local_size = 1;
    }else{
        cl_device_id raw_device = ctx.current_device().id();
        cl_kernel raw_kernel = ctx.get_kernel("my_kernel", "MatSign").handle().get();
        size_t preferred_work_group_size_multiple;
        
        cl_int err = clGetKernelWorkGroupInfo(raw_kernel, raw_device, 
                                              CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                              sizeof(size_t), &preferred_work_group_size_multiple, NULL);
        
        if(err != CL_SUCCESS){
            Rcpp::stop("clGetKernelWorkGroupInfo failed");
        }
        
        max_local_size = roundDown(max_local_size, preferred_work_group_size_multiple);
    }
    
    // set global work sizes
    mat_sign.global_work_size(0, M_internal);
    mat_sign.global_work_size(1, P_internal);
    
    // set local work sizes
    mat_sign.local_work_size(0, max_local_size);
    mat_sign.local_work_size(1, max_local_size);
    
    // execute kernels
    for(unsigned int k=0; k < M; k++){
        viennacl::ocl::enqueue(mat_sign(*vcl_A, *vcl_B, M, P, M_internal));
    }
    
    if(!BisVCL){
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        
        // copy device data back to CPU
        ptrB->to_host(*vcl_B);
        ptrB->release_device();
    }
}

template<typename T>
void
cpp_vclMatrix_pmax(
    SEXP ptrA_,
    const bool AisVCL,
    SEXP ptrB_, 
    const bool BisVCL,
    SEXP value_,
    SEXP sourceCode_,
    unsigned int max_local_size,
    const int ctx_id)
{
    std::string my_kernel = as<std::string>(sourceCode_);
    
    viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    T value = as<T>(value_);
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
    
    unsigned int M = vcl_B->size1();
    unsigned int P = vcl_B->size2();
    unsigned int M_internal = vcl_B->internal_size1();
    unsigned int P_internal = vcl_B->internal_size2();
    
    // add kernel to program
    viennacl::ocl::program & my_prog = ctx.add_program(my_kernel, "my_kernel");
    
    // get compiled kernel function
    viennacl::ocl::kernel & pmax = my_prog.get_kernel("pmax");
    
    cl_device_type type_check = ctx.current_device().type();
    
    if(type_check & CL_DEVICE_TYPE_CPU){
        max_local_size = 1;
    }else{
        cl_device_id raw_device = ctx.current_device().id();
        cl_kernel raw_kernel = ctx.get_kernel("my_kernel", "pmax").handle().get();
        size_t preferred_work_group_size_multiple;
        
        cl_int err = clGetKernelWorkGroupInfo(raw_kernel, raw_device, 
                                              CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                              sizeof(size_t), &preferred_work_group_size_multiple, NULL);
        
        if(err != CL_SUCCESS){
            Rcpp::stop("clGetKernelWorkGroupInfo failed");
        }
        
        max_local_size = roundDown(max_local_size, preferred_work_group_size_multiple);
    }
    
    // set global work sizes
    const unsigned int globalSize1 = roundUp(M_internal, max_local_size);
    const unsigned int globalSize2 = roundUp(P_internal, max_local_size);
    pmax.global_work_size(0, globalSize1);
    pmax.global_work_size(1, globalSize2);
    
    // set local work sizes
    pmax.local_work_size(0, max_local_size);
    pmax.local_work_size(1, max_local_size);
    
    // execute kernels
    for(unsigned int k=0; k < M; k++){
        viennacl::ocl::enqueue(pmax(*vcl_A, *vcl_B, value, M, P, M_internal));
    }
    
    if(!BisVCL){
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        
        // copy device data back to CPU
        ptrB->to_host(*vcl_B);
        ptrB->release_device();
    }
}


template<typename T>
void
cpp_vclVector_sign(
    SEXP ptrA_,
    const bool AisVCL,
    SEXP ptrB_, 
    const bool BisVCL,
    SEXP sourceCode_,
    unsigned int max_local_size,
    const int ctx_id)
{
    std::string my_kernel = as<std::string>(sourceCode_);
    
    viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_B = getVCLVecptr<T>(ptrB_, BisVCL, ctx_id);
    
    unsigned int M = vcl_B->size();

    // add kernel to program
    viennacl::ocl::program & my_prog = ctx.add_program(my_kernel, "my_kernel");
    
    // get compiled kernel function
    viennacl::ocl::kernel & mat_sign = my_prog.get_kernel("VecSign");
    
    cl_device_type type_check = ctx.current_device().type();
    
    if(type_check & CL_DEVICE_TYPE_CPU){
        max_local_size = 1;
    }else{
        cl_device_id raw_device = ctx.current_device().id();
        cl_kernel raw_kernel = ctx.get_kernel("my_kernel", "VecSign").handle().get();
        size_t preferred_work_group_size_multiple;
        
        cl_int err = clGetKernelWorkGroupInfo(raw_kernel, raw_device, 
                                              CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                              sizeof(size_t), &preferred_work_group_size_multiple, NULL);
        
        if(err != CL_SUCCESS){
            Rcpp::stop("clGetKernelWorkGroupInfo failed");
        }
        
        max_local_size = roundDown(max_local_size, preferred_work_group_size_multiple);
    }
    
    // set global work sizes
    int globalSize = roundUp(M, max_local_size);
    mat_sign.global_work_size(0, globalSize);
    
    // set local work sizes
    mat_sign.local_work_size(0, max_local_size);
    
    // execute kernels
    viennacl::ocl::enqueue(mat_sign(*vcl_A, *vcl_B, M));
    
    if(!BisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrB(ptrB_);
        
        // copy device data back to CPU
        ptrB->to_host(*vcl_B);
        ptrB->release_device();
    }
}


template<typename T>
void
cpp_vclVector_pmax(
    SEXP ptrA_,
    const bool AisVCL,
    SEXP ptrB_, 
    const bool BisVCL,
    SEXP value_,
    SEXP sourceCode_,
    unsigned int max_local_size,
    const int ctx_id)
{
    std::string my_kernel = as<std::string>(sourceCode_);
    
    viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    T value = as<T>(value_);
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_B = getVCLVecptr<T>(ptrB_, BisVCL, ctx_id);
    
    unsigned int M = vcl_B->size();
    
    // add kernel to program
    viennacl::ocl::program & my_prog = ctx.add_program(my_kernel, "my_kernel");
    
    // get compiled kernel function
    viennacl::ocl::kernel & pmax = my_prog.get_kernel("pmax");
    
    cl_device_type type_check = ctx.current_device().type();
    
    if(type_check & CL_DEVICE_TYPE_CPU){
        max_local_size = 1;
    }else{
        cl_device_id raw_device = ctx.current_device().id();
        cl_kernel raw_kernel = ctx.get_kernel("my_kernel", "pmax").handle().get();
        size_t preferred_work_group_size_multiple;
        
        cl_int err = clGetKernelWorkGroupInfo(raw_kernel, raw_device, 
                                              CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                              sizeof(size_t), &preferred_work_group_size_multiple, NULL);
        
        if(err != CL_SUCCESS){
            Rcpp::stop("clGetKernelWorkGroupInfo failed");
        }
        
        max_local_size = roundDown(max_local_size, preferred_work_group_size_multiple);
    }
    
    // set global work sizes
    int globalSize = roundUp(M, max_local_size);
    pmax.global_work_size(0, globalSize);
    
    // set local work sizes
    pmax.local_work_size(0, max_local_size);
    
    // execute kernels
    viennacl::ocl::enqueue(pmax(*vcl_A, *vcl_B, value, M));
    
    if(!BisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrB(ptrB_);
        
        // copy device data back to CPU
        ptrB->to_host(*vcl_B);
        ptrB->release_device();
    }
}



// [[Rcpp::export]]
void
cpp_vclMatrix_sign(
    SEXP ptrA,
    const bool AisVCL,
    SEXP ptrB, 
    const bool BisVCL,
    SEXP sourceCode,
    int max_local_size,
    const int type_flag,
    const int ctx_id)
{
    switch(type_flag) {
    case 4:
        cpp_vclMatrix_sign<int>(ptrA, AisVCL, ptrB, BisVCL, sourceCode, max_local_size, ctx_id);
        return;
    case 6:
        cpp_vclMatrix_sign<float>(ptrA, AisVCL, ptrB, BisVCL, sourceCode, max_local_size, ctx_id);
        return;
    case 8:
        cpp_vclMatrix_sign<double>(ptrA, AisVCL, ptrB, BisVCL, sourceCode, max_local_size, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclVector_sign(
    SEXP ptrA,
    const bool AisVCL,
    SEXP ptrB, 
    const bool BisVCL,
    SEXP sourceCode,
    int max_local_size,
    const int type_flag,
    const int ctx_id)
{
    switch(type_flag) {
    case 4:
        cpp_vclVector_sign<int>(ptrA, AisVCL, ptrB, BisVCL, sourceCode, max_local_size, ctx_id);
        return;
    case 6:
        cpp_vclVector_sign<float>(ptrA, AisVCL, ptrB, BisVCL, sourceCode, max_local_size, ctx_id);
        return;
    case 8:
        cpp_vclVector_sign<double>(ptrA, AisVCL, ptrB, BisVCL, sourceCode, max_local_size, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclMatrix_pmax(
    SEXP ptrA,
    const bool AisVCL,
    SEXP ptrB, 
    const bool BisVCL,
    SEXP value,
    SEXP sourceCode,
    int max_local_size,
    const int type_flag,
    const int ctx_id)
{
    switch(type_flag) {
    case 4:
        cpp_vclMatrix_pmax<int>(ptrA, AisVCL, ptrB, BisVCL, value, sourceCode, max_local_size, ctx_id);
        return;
    case 6:
        cpp_vclMatrix_pmax<float>(ptrA, AisVCL, ptrB, BisVCL, value, sourceCode, max_local_size, ctx_id);
        return;
    case 8:
        cpp_vclMatrix_pmax<double>(ptrA, AisVCL, ptrB, BisVCL, value, sourceCode, max_local_size, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_pmax(
    SEXP ptrA,
    const bool AisVCL,
    SEXP ptrB, 
    const bool BisVCL,
    SEXP value,
    SEXP sourceCode,
    int max_local_size,
    const int type_flag,
    const int ctx_id)
{
    switch(type_flag) {
    case 4:
        cpp_vclVector_pmax<int>(ptrA, AisVCL, ptrB, BisVCL, value, sourceCode, max_local_size, ctx_id);
        return;
    case 6:
        cpp_vclVector_pmax<float>(ptrA, AisVCL, ptrB, BisVCL, value, sourceCode, max_local_size, ctx_id);
        return;
    case 8:
        cpp_vclVector_pmax<double>(ptrA, AisVCL, ptrB, BisVCL, value, sourceCode, max_local_size, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}
