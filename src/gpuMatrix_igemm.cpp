
#include "gpuR/windows_check.hpp"

#include <RcppEigen.h>

#include "viennacl/ocl/backend.hpp"

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
        const int max_local_size,
        const int ctx_id)
{
    std::string my_kernel = as<std::string>(sourceCode_);
    
    viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    viennacl::matrix<int> *vcl_A;
    viennacl::matrix<int> *vcl_B;
    viennacl::matrix<int> *vcl_C;
    
    vcl_A = getVCLptr<int>(ptrA_, AisVCL, ctx_id);
    vcl_B = getVCLptr<int>(ptrB_, BisVCL, ctx_id);
    vcl_C = getVCLptr<int>(ptrC_, CisVCL, ctx_id);
    
    int M = vcl_A->size1();
    // int N = vcl_B.size1();
    int P = vcl_B->size2();
    int M_internal = vcl_C->internal_size2();
    int P_internal = vcl_C->internal_size1();
    
    // add kernel to program
    viennacl::ocl::program & my_prog = ctx.add_program(my_kernel, "my_kernel");
    
    // get compiled kernel function
    viennacl::ocl::kernel & my_kernel_mul = my_prog.get_kernel("iMatMult");

    // set global work sizes
    my_kernel_mul.global_work_size(0, M_internal);
    my_kernel_mul.global_work_size(1, P_internal);
    
    // set local work sizes
    my_kernel_mul.local_work_size(0, max_local_size);
    my_kernel_mul.local_work_size(1, max_local_size);
    
    // execute kernel
    viennacl::ocl::enqueue(my_kernel_mul(M, M_internal, P, P_internal, *vcl_A, *vcl_B, *vcl_C));
    
    if(!CisVCL){
        // move back to host
        Rcpp::XPtr<dynEigenMat<int> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device(); 
    }
}

