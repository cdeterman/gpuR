#include "gpuR/windows_check.hpp"

#include <RcppEigen.h>

#include "viennacl/ocl/backend.hpp"

#include "gpuR/utils.hpp"
#include "gpuR/getVCLptr.hpp"

// using namespace cl;
using namespace Rcpp;

template<typename T>
void
cpp_vclVector_permute(
    SEXP ptrA_, 
    Eigen::VectorXi indices,
    SEXP sourceCode_,
    const int ctx_id)
{
    std::string my_kernel = as<std::string>(sourceCode_);
    
    viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    Rcpp::XPtr<dynVCLVec<T> > pA(ptrA_);
    viennacl::vector_range<viennacl::vector_base<T> > vcl_A  = pA->data();
    
    viennacl::vector<T> vcl_B = viennacl::zero_vector<T>(vcl_A.size());
    
    // copy indices to device
    viennacl::vector<int> vcl_I(indices.size());
    viennacl::copy(indices, vcl_I);
    
    // std::cout << vcl_A.size() << std::endl;
    // std::cout << vcl_A.internal_size() << std::endl;
        
    // // add kernel to program
    // viennacl::ocl::program & my_prog = ctx.add_program(my_kernel, "my_kernel");
    // 
    // // get compiled kernel function
    // viennacl::ocl::kernel & permute = my_prog.get_kernel("dPermute");
    // 
    // // set global work sizes
    // permute.global_work_size(0, 1);
    // 
    // // execute kernel
    // viennacl::ocl::enqueue(permute(vcl_A, vcl_B, vcl_I));
    // 
    // std::cout << vcl_B << std::endl;
}

template<typename T>
void
cpp_vclMatrix_set_row_order(
    SEXP ptrA_, 
    const bool AisVCL,
    Eigen::VectorXi indices,
    SEXP sourceCode_,
    int max_local_size,
    const int ctx_id)
{
    
    // std::cout << "called" << std::endl;
    
    std::string my_kernel = as<std::string>(sourceCode_);
    
    viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    // Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    // viennacl::matrix<T> vcl_A = ptrA->data();
    // viennacl::matrix<T> *vcl_B;
    
    // std::cout << "getting matrix" << std::endl;
    std::shared_ptr<viennacl::matrix_range<viennacl::matrix<T> > > vcl_A = getVCLBlockptr<T>(ptrA_, AisVCL, ctx_id);
    // vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
    
    unsigned int M = vcl_A->size1();
    // // int N = vcl_B.size1();
    unsigned int P = vcl_A->size2();
    unsigned int M_internal = vcl_A->internal_size1();
    unsigned int P_internal = vcl_A->internal_size2();
    
    // std::cout << M_internal << std::endl;
    
    // std::cout << "initialized" << std::endl;
    
    viennacl::ocl::kernel set_row_order;
    
    // try {
    //     // protected code
    //     
    //     // add kernel to program
    //     viennacl::ocl::program & my_prog = ctx.get_program("permute_kernel");
    //     
    //     // std::cout << "got program" << std::endl;
    //     
    //     // get compiled kernel function
    //     set_row_order = my_prog.get_kernel("set_row_order");
    //     
    //     // std::cout << "got kernel" << std::endl;
    // }catch(...) {
        // code to handle any exception
        // std::cout << "exception handler" << std::endl;
        // add kernel to program
        viennacl::ocl::program & my_prog = ctx.add_program(my_kernel, "permute_kernel");
        
        // std::cout << "program added" << std::endl;
        
        // get compiled kernel function
        set_row_order = my_prog.get_kernel("set_row_order");
        
        // std::cout << "got kernel" << std::endl;
    // }
    
    cl_device_type type_check = ctx.current_device().type();
    
    if(type_check & CL_DEVICE_TYPE_CPU){
        max_local_size = 1;
    }else{
        cl_device_id raw_device = ctx.current_device().id();
        cl_kernel raw_kernel = ctx.get_kernel("permute_kernel", "set_row_order").handle().get();
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
    set_row_order.global_work_size(0, M_internal);
    // set_row_order.global_work_size(1, P_internal);
    
    // std::cout << "set global" << std::endl;
    
    // set local work sizes
    set_row_order.local_work_size(0, max_local_size);
    // set_row_order.local_work_size(1, max_local_size);
    
    // std::cout << "begin enqueue" << std::endl;
    
    // std::cout << vcl_I << std::endl;
    // 
    // std::cout << *vcl_A << std::endl;
    // std::cout << *vcl_B << std::endl;
    
    {
        
        // std::cout << "moving indexes" << std::endl;
        viennacl::vector<int> vcl_I(indices.size());
        viennacl::copy(indices, vcl_I);
        
        // std::cout << "creating dummy vector" << std::endl;
        viennacl::vector<T> vcl_V = viennacl::zero_vector<T>(M_internal);
        
        viennacl::matrix_base<T> vcl_B(vcl_V.handle(),
                                       M_internal, 0, 1, M_internal,   //row layout
                                       1, 0, 1, 1,   //column layout
                                       true); // row-major
        
        viennacl::range r(0, M);
        
        for(unsigned int i=0; i < P; i++){
            
            // std::cout << "column: " << i << std::endl;
            
            viennacl::range c(i, i+1);
            
            viennacl::matrix_range<viennacl::matrix<T> > tmp(*vcl_A, r, c);
            
            // std::cout << tmp << std::endl;
            
            viennacl::ocl::enqueue(set_row_order(tmp, vcl_B, vcl_I, M, i, P_internal));
            
            tmp = vcl_B;
        }
    }
    
    // delete vcl_A;
    
    // // execute kernel
    // viennacl::ocl::enqueue(set_row_order(*vcl_A, *vcl_B, vcl_I, M, P, M_internal));
    
    // std::cout << "enqueue finished" << std::endl;
    
    // if(!BisVCL){
    //     Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
    //     
    //     // copy device data back to CPU
    //     ptrB->to_host(*vcl_B);
    //     ptrB->release_device();
    // }
    
    // std::cout << *vcl_B << std::endl;
}


// [[Rcpp::export]]
void
cpp_vclVector_permute(
    SEXP ptrA, 
    Eigen::VectorXi indices,
    SEXP sourceCode,
    const int type_flag,
    const int ctx_id)
{
    switch(type_flag) {
    case 6:
        cpp_vclVector_permute<float>(ptrA, indices, sourceCode, ctx_id);
        return;
    case 8:
        cpp_vclVector_permute<double>(ptrA, indices, sourceCode, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}
    
// [[Rcpp::export]]
void
cpp_vclMatrix_set_row_order(
    SEXP ptrA, 
    const bool AisVCL,
    Eigen::VectorXi indices,
    SEXP sourceCode,
    const int max_local_size,
    const int type_flag,
    const int ctx_id)
{
    switch(type_flag) {
    case 6:
        cpp_vclMatrix_set_row_order<float>(ptrA, AisVCL, indices, sourceCode, max_local_size, ctx_id);
        return;
    case 8:
        cpp_vclMatrix_set_row_order<double>(ptrA, AisVCL, indices, sourceCode, max_local_size, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


