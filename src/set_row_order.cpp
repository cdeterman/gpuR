#include "gpuR/windows_check.hpp"

#include <RcppEigen.h>

#include "viennacl/ocl/backend.hpp"

#include "gpuR/getVCLptr.hpp"

// using namespace cl;
using namespace Rcpp;

template<typename T>
void
    cpp_vclMatrix_set_row_order(
        SEXP ptrA_, 
        SEXP ptrB_, 
        const bool AisVCL,
        const bool BisVCL,
        Eigen::VectorXi indices,
        SEXP sourceCode_,
        const int max_local_size,
        const int ctx_id)
    {
        std::string my_kernel = as<std::string>(sourceCode_);
        
        viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
        
        viennacl::matrix<T> *vcl_A;
        viennacl::matrix<T> *vcl_B;
        
        viennacl::vector<int> vcl_I(indices.size());
        viennacl::copy(indices, vcl_I);
        
        vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
        vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
        
        unsigned int M = vcl_A->size1();
        // // int N = vcl_B.size1();
        unsigned int P = vcl_A->size2();
        unsigned int M_internal = vcl_A->internal_size1();
        unsigned int P_internal = vcl_A->internal_size2();
        
        // std::cout << "initialized" << std::endl;
        
        // add kernel to program
        viennacl::ocl::program & my_prog = ctx.add_program(my_kernel, "my_kernel");
        
        // get compiled kernel function
        viennacl::ocl::kernel & set_row_order = my_prog.get_kernel("set_row_order");
        
        // std::cout << "got kernel" << std::endl;
        
        // set global work sizes
        set_row_order.global_work_size(0, M_internal);
        set_row_order.global_work_size(1, P_internal);
        
        // set local work sizes
        set_row_order.local_work_size(0, max_local_size);
        set_row_order.local_work_size(1, max_local_size);
        
        // std::cout << "begin enqueue" << std::endl;
        
        // std::cout << vcl_I << std::endl;
        // 
        // std::cout << *vcl_A << std::endl;
        // std::cout << *vcl_B << std::endl;
        
        // execute kernel
        viennacl::ocl::enqueue(set_row_order(*vcl_A, *vcl_B, vcl_I, M, P, M_internal));
        
        // std::cout << "enqueue finished" << std::endl;
        
        if(!BisVCL){
            Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
            
            // copy device data back to CPU
            ptrB->to_host(*vcl_B);
            ptrB->release_device();
        }
        
        // std::cout << *vcl_B << std::endl;
    }



// [[Rcpp::export]]
void
cpp_vclMatrix_set_row_order(
    SEXP ptrA, 
    SEXP ptrB, 
    const bool AisVCL,
    const bool BisVCL,
    Eigen::VectorXi indices,
    SEXP sourceCode,
    const int max_local_size,
    const int type_flag,
    const int ctx_id)
{
    switch(type_flag) {
    case 6:
        cpp_vclMatrix_set_row_order<float>(ptrA, ptrB, AisVCL, BisVCL, indices, sourceCode, max_local_size, ctx_id);
        return;
    case 8:
        cpp_vclMatrix_set_row_order<double>(ptrA, ptrB, AisVCL, BisVCL, indices, sourceCode, max_local_size, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


