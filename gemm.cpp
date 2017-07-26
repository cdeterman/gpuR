// [[Rcpp::depends(BH, RcppEigen, RViennaCL, gpuR)]]
// [[Rcpp::plugins(cpp11)]]
#include "gpuR/windows_check.hpp"

#include <RcppEigen.h>

// #define VIENNACL_DEBUG_ALL 1

#include "viennacl/ocl/backend.hpp"

#include "gpuR/utils.hpp"
#include "gpuR/getVCLptr.hpp"

using namespace Rcpp;
// 
// void initContexts(){
//     // declarations
//     int id = 0;
//     
//     // get platforms
//     typedef std::vector< viennacl::ocl::platform > platforms_type;
//     platforms_type platforms = viennacl::ocl::get_platforms();
//     
//     Rcpp::Rcout << "Number of platforms: " << platforms.size() << std::endl;
//     
//     for(unsigned int plat_idx = 0; plat_idx < platforms.size(); plat_idx++) {
//         
//         Rcpp::Rcout << "- platform: " << platforms[plat_idx].info() << std::endl;
//         
//         std::vector< viennacl::ocl::device > devices;
//         devices = platforms[plat_idx].devices(CL_DEVICE_TYPE_ALL);
//         
//         for(unsigned int gpu_idx = 0; gpu_idx < devices.size(); gpu_idx++) {
//             
//             Rcpp::Rcout << "  - gpu index: " << gpu_idx << std::endl;
//             viennacl::ocl::set_context_platform_index(id, plat_idx);
//             viennacl::ocl::setup_context(id, devices[gpu_idx]);
//             Rcpp::Rcout << "    - " << devices[gpu_idx].name() << std::endl;
//             
//             // increment context
//             id++;
//         }
//     }
//     
//     Rcpp::Rcout << "checked all devices" << std::endl;
//     
//     viennacl::ocl::switch_context(0);
//     
//     Rcpp::Rcout << "completed initialization" << std::endl;
// }

//[[Rcpp::export]]
void 
    cpp_gpuMatrix_custom_igemm2(
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
        
        // initContexts();
        
        // std::cout << viennacl::ocl::backend<>::current_context_id() << std::endl;
        
        viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
        // Rcpp::XPtr<viennacl::ocl::context> ctx(ctx_ptr);
        
        // std::cout << ctx_id << std::endl;
        // std::cout  << ctx << std::endl;
        
        
        std::shared_ptr<viennacl::matrix<int> > vcl_A = getVCLptr<int>(ptrA_, AisVCL, ctx_id);
        std::shared_ptr<viennacl::matrix<int> > vcl_B = getVCLptr<int>(ptrB_, BisVCL, ctx_id);
        std::shared_ptr<viennacl::matrix<int> > vcl_C = getVCLptr<int>(ptrC_, CisVCL, ctx_id);
        
        std::cout << *vcl_A << std::endl;
        std::cout << *vcl_C << std::endl;
            
        int M = vcl_A->size2();
        // int N = vcl_B.size1();
        int P = vcl_B->size1();
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
            
            max_local_size = roundDown(max_local_size, preferred_work_group_size_multiple);
        }
        
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