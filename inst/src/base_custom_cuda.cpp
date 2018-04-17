// [[Rcpp::depends(BH, RcppEigen, RViennaCL, gpuR)]]
// [[Rcpp::plugins(cpp11)]]

#include "gpuR/windows_check.hpp"

#include <RcppEigen.h>

#include "viennacl/ocl/backend.hpp"

#include "gpuR/utils.hpp"
#include "gpuR/getVCLptr.hpp"

using namespace Rcpp;

template<typename T>
__global__ void MY_KERNEL
    

// [[Rcpp::export]]
void
    CPP_NAME(
        MY_ARGS)
    {
        
        MY_KERNEL_NAMES
        
        MY_S4
            
        MY_CTX_ID
        
        MY_DEFINES
        
        MY_CONTEXT
        
        MY_DIMS

        // add kernel to program
        viennacl::ocl::program & my_prog = ctx.add_program(my_kernel, "my_kernel");
        
        // get compiled kernel function
        MY_KERNELS

        viennacl::ocl::device working_device = ctx.current_device();
        
        // set global work sizes
        MY_GLOBALS

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

