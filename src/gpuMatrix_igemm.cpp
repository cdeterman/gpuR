
#include "gpuR/windows_check.hpp"

#include <RcppEigen.h>

#include "viennacl/ocl/backend.hpp"

#include "gpuR/dynEigenMat.hpp"
#include "gpuR/cl_helpers.hpp"

using namespace cl;
using namespace Rcpp;


//[[Rcpp::export]]
void 
cpp_gpuMatrix_custom_igemm(
        SEXP ptrA_, SEXP ptrB_, SEXP ptrC_,
        SEXP sourceCode_,
        int max_local_size,
        int ctx_id)
{
    std::string my_kernel = as<std::string>(sourceCode_);
    
    XPtr<dynEigenMat<int> > ptrA(ptrA_);
    XPtr<dynEigenMat<int> > ptrB(ptrB_);
    XPtr<dynEigenMat<int> > ptrC(ptrC_);
    
    // move data to device
    viennacl::matrix<int> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<int> vcl_B = ptrB->device_data(ctx_id);
    viennacl::matrix<int> vcl_C = ptrC->device_data(ctx_id);
    
    int M = vcl_A.size1();
    // int N = vcl_B.size1();
    int P = vcl_B.size2();
    int M_internal = vcl_C.internal_size2();
    int P_internal = vcl_C.internal_size1();
    
    // add kernel to program
    viennacl::ocl::program & my_prog = viennacl::ocl::current_context().add_program(my_kernel, "my_kernel");
    
    // get compiled kernel function
    viennacl::ocl::kernel & my_kernel_mul = my_prog.get_kernel("iMatMult");

    // set global work sizes
    my_kernel_mul.global_work_size(0, M_internal);
    my_kernel_mul.global_work_size(1, P_internal);
    
    // set local work sizes
    my_kernel_mul.local_work_size(0, max_local_size);
    my_kernel_mul.local_work_size(1, max_local_size);
    
    // execute kernel
    viennacl::ocl::enqueue(my_kernel_mul(M, M_internal, P, P_internal, vcl_A, vcl_B, vcl_C));
    
    // move back to host
    ptrC->to_host(vcl_C);
}

