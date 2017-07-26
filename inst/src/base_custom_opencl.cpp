// [[Rcpp::depends(BH, RcppEigen, RViennaCL, gpuR)]]
// [[Rcpp::plugins(cpp11)]]

#include "gpuR/windows_check.hpp"

#include <RcppEigen.h>

#include "viennacl/ocl/backend.hpp"

#include "gpuR/utils.hpp"
#include "gpuR/getVCLptr.hpp"

using namespace Rcpp;

// [[Rcpp::export]]
void
CPP_NAME(
    MY_ARGS)
{
    
    MY_KERNEL_NAMES
    
    const char * my_kernel = MY_KERNEL_SRC
    ;
    
    MY_S4
        
    MY_CTX_ID
    
    MY_DEFINES
    // T value = as<T>(value_);
    // std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    // std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);
    
    MY_CONTEXT
        
    MY_DIMS
    // unsigned int M = vcl_B->size1();
    // unsigned int P = vcl_B->size2();
    // unsigned int M_internal = vcl_B->internal_size1();
    // unsigned int P_internal = vcl_B->internal_size2();
    
    // add kernel to program
    viennacl::ocl::program & my_prog = ctx.add_program(my_kernel, "my_kernel");
    
    // get compiled kernel function
    MY_KERNELS
    // viennacl::ocl::kernel & update_kk = my_prog.get_kernel("update_kk");
    // viennacl::ocl::kernel & update_k = my_prog.get_kernel("update_k");
    // viennacl::ocl::kernel & update_block = my_prog.get_kernel("update_block");
    // viennacl::ocl::kernel & my_kernel = my_prog.get_kernel(kernel_name);
    
    viennacl::ocl::device working_device = ctx.current_device();
    Rcpp::IntegerVector max_local_size(kernel_name.size(), working_device.max_work_group_size());
        
    cl_device_type type_check = working_device.type();
    
    for(unsigned int i = 0; i < kernel_name.size(); i++){
        if(type_check & CL_DEVICE_TYPE_CPU){
            max_local_size[i] = 1;
        }else{
            cl_device_id raw_device = working_device.id();
            cl_kernel raw_kernel = ctx.get_kernel("my_kernel", as<std::string>(kernel_name[i])).handle().get();
            size_t preferred_work_group_size_multiple;
            
            cl_int err = clGetKernelWorkGroupInfo(raw_kernel, raw_device, 
                                                  CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                                  sizeof(size_t), &preferred_work_group_size_multiple, NULL);
            
            if(err != CL_SUCCESS){
                Rcpp::stop("clGetKernelWorkGroupInfo failed");
            }
            
            max_local_size[i] = roundDown(max_local_size[i], preferred_work_group_size_multiple);
        }
    }
    
    
    // set global work sizes
    MY_GLOBALS
    // const unsigned int globalSize1 = roundUp(M_internal, max_local_size);
    // const unsigned int globalSize2 = roundUp(P_internal, max_local_size);
    // pmax.global_work_size(0, globalSize1);
    // pmax.global_work_size(1, globalSize2);
    
    // set local work sizes
    MY_LOCALS
    // pmax.local_work_size(0, max_local_size);
    // pmax.local_work_size(1, max_local_size);
    
    // execute kernels
    MY_QUEUES
    // viennacl::ocl::enqueue(my_kernel(*vcl_A, *vcl_B, value, M, P, M_internal));
    
    // device to host if needed
    MY_OUT
}

