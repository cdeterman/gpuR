
#include "gpuR/windows_check.hpp"

#include <RcppEigen.h>

#include "viennacl/ocl/backend.hpp"

#include "gpuR/utils.hpp"
#include "gpuR/getVCLptr.hpp"

using namespace Rcpp;

//[[Rcpp::export]]
void 
cpp_gpuMatrix_custom_igemm(
        SEXP ptrA_, 
        const bool AisVCL,
        SEXP ptrB_, 
        const bool BisVCL,
        SEXP ptrC_,
        const bool CisVCL,
        SEXP sourceCode_,
        int max_local_size,
        const int ctx_id)
{
    std::string my_kernel = as<std::string>(sourceCode_);
    
    viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    std::shared_ptr<viennacl::matrix_range<viennacl::matrix<int> > > vcl_A = getVCLBlockptr<int>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::matrix_range<viennacl::matrix<int> > > vcl_B = getVCLBlockptr<int>(ptrB_, BisVCL, ctx_id);
    std::shared_ptr<viennacl::matrix_range<viennacl::matrix<int> > > vcl_C = getVCLBlockptr<int>(ptrC_, CisVCL, ctx_id);
    
    int M = vcl_A->size1();
    int N = vcl_B->size2();
    int P = vcl_B->size1();
    // int A_rows = vcl_A->internal_size1();
    int A_cols = vcl_A->internal_size2();
    int B_cols = vcl_B->internal_size2();
    int M_internal = vcl_C->internal_size2();
    int P_internal = vcl_C->internal_size1();
    
    // add kernel to program
    viennacl::ocl::program & my_prog = ctx.add_program(my_kernel, "my_kernel");
    
    // get compiled kernel function
    viennacl::ocl::kernel & my_kernel_mul = my_prog.get_kernel("iMatMult");
    
    cl_device_type type_check = ctx.current_device().type();
    
    if(type_check & CL_DEVICE_TYPE_CPU){
        max_local_size = 1;
    }else{
        cl_device_id raw_device = ctx.current_device().id();
        cl_kernel raw_kernel = ctx.get_kernel("my_kernel", "iMatMult").handle().get();
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
    my_kernel_mul.global_work_size(0, P_internal);
    my_kernel_mul.global_work_size(1, M_internal);
    
    // set local work sizes
    my_kernel_mul.local_work_size(0, max_local_size);
    my_kernel_mul.local_work_size(1, max_local_size);
    
    // execute kernel
    viennacl::ocl::enqueue(my_kernel_mul(M, A_cols, P, N, B_cols, M_internal, *vcl_A, *vcl_B, *vcl_C));
    
    if(!CisVCL){
        // move back to host
        Rcpp::XPtr<dynEigenMat<int> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device(); 
    }
}

