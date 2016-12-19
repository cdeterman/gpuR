#include "gpuR/windows_check.hpp"

// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/utils.hpp"
#include "gpuR/getVCLptr.hpp"
// #include "gpuR/dynEigenMat.hpp"
#include "gpuR/dynEigenVec.hpp"
// #include "gpuR/dynVCLMat.hpp"
#include "gpuR/dynVCLVec.hpp"

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

// ViennaCL headers
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/maxmin.hpp"


template <typename T>
void cpp_gpu_rprop_plus(
        SEXP ptrGradients_,
        SEXP ptrGradientsOld_,
        SEXP ptrWeights_,
        SEXP ptrlr_,
        Rcpp::List learningrate_factor,
        Rcpp::List learningrate_limit,
        SEXP sourceCode_,
        int max_local_size,
        const int ctx_id)
{
    
    viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
    std::string my_kernel = Rcpp::as<std::string>(sourceCode_);
    
    Rcpp::XPtr<dynVCLVec<T> > ptrGradients(ptrGradients_);
    Rcpp::XPtr<dynVCLVec<T> > ptrGradientsOld(ptrGradientsOld_);
    Rcpp::XPtr<dynVCLVec<T> > ptrWeights(ptrWeights_);
    Rcpp::XPtr<dynVCLVec<T> > ptrlr(ptrlr_);
    
    viennacl::vector_range<viennacl::vector_base<T> > gradients = ptrGradients->data();
    viennacl::vector_range<viennacl::vector_base<T> > gradients_old = ptrGradientsOld->data();
    viennacl::vector_range<viennacl::vector_base<T> > weights = ptrWeights->data();
    viennacl::vector_range<viennacl::vector_base<T> > learningrate = ptrlr->data();
    
    const T lr_plus = Rcpp::as<T>(learningrate_factor["plus"]);
    const T lr_minus = Rcpp::as<T>(learningrate_factor["minus"]);
    const T lr_max = Rcpp::as<T>(learningrate_limit["max"]);
    const T lr_min = Rcpp::as<T>(learningrate_limit["min"]);
    const int M = gradients.size();
    
    // std::cout << "length: " << M << std::endl;
    
    // add kernel to program
    viennacl::ocl::program & my_prog = ctx.add_program(my_kernel, "my_kernel");
    
    // get compiled kernel function
    viennacl::ocl::kernel & my_kernel_mul = my_prog.get_kernel("rprop_plus");
    
    cl_device_type type_check = ctx.current_device().type();
    
    if(type_check & CL_DEVICE_TYPE_CPU){
        max_local_size = 1;
    }else{
        cl_device_id raw_device = ctx.current_device().id();
        cl_kernel raw_kernel = ctx.get_kernel("my_kernel", "rprop_plus").handle().get();
        size_t preferred_work_group_size_multiple;
        
        cl_int err = clGetKernelWorkGroupInfo(raw_kernel, raw_device, 
                                              CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                              sizeof(size_t), &preferred_work_group_size_multiple, NULL);
        
        max_local_size = roundDown(max_local_size, preferred_work_group_size_multiple);
    }
    
    // std::cout << "local size" << std::endl;
    // std::cout << max_local_size << std::endl;
    // std::cout << "gradient size" << std::endl;
    // std::cout << gradients.size() << std::endl;
    // 
    // set global work sizes
    if(gradients.size() < max_local_size){
        my_kernel_mul.global_work_size(0, max_local_size);
    }else{
    		my_kernel_mul.global_work_size(0, roundUp(gradients.size(), max_local_size));
    }

    // set local work sizes
    my_kernel_mul.local_work_size(0, max_local_size);
    
    // execute kernel
    viennacl::ocl::enqueue(my_kernel_mul(gradients, gradients_old, weights, learningrate, 
                                         lr_plus, lr_minus, lr_max, lr_min, M));
}


// [[Rcpp::export]]
void
cpp_gpu_rprop_plus(
    SEXP ptrGradients,
    SEXP ptrGradientsOld,
    SEXP ptrWeights,
    SEXP ptrlr,
    Rcpp::List learningrate_factor,
    Rcpp::List learningrate_limit,
    SEXP sourceCode,
    int max_local_size,
    const int ctx_id,
    const int type_flag)
{
    
    switch(type_flag) {
    case 4:
        throw Rcpp::exception("integer type not implemented");
    case 6:
        cpp_gpu_rprop_plus<float>(ptrGradients, ptrGradientsOld, ptrWeights, ptrlr,
                                  learningrate_factor, learningrate_limit,
                                  sourceCode, max_local_size, ctx_id);
        return;
    case 8:
        cpp_gpu_rprop_plus<double>(ptrGradients, ptrGradientsOld, ptrWeights, ptrlr,
                                   learningrate_factor, learningrate_limit,
                                   sourceCode, max_local_size, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

