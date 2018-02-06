
#include "gpuR/windows_check.hpp"
#include "gpuR/utils.hpp"
#include "gpuR/getVCLptr.hpp"

using namespace Rcpp;

// custom kernels for CUDA backend
#ifdef BACKEND_CUDA

__global__ void iMatMult(const int A_size2, const int C_internal_size2, 
                       const int B_size1, const int C_internal_size1,
                       const int *A, const int *B, int *C) {
    
    // Get the index of the elements to be processed
    const int globalRow = threadIdx.x; // C Row ID
    const int globalCol = threadIdx.y; // C Col ID
    int tmp = 0;
    
    // Do the operation
    if((globalRow <= A_size2) && (globalCol <= B_size1)){
        
        for(int k=0; k < B_size1; k++){
            tmp += A[globalRow * C_internal_size2 + k] * B[globalCol+C_internal_size1*k];
        }
        
        C[globalCol+C_internal_size2*globalRow] = tmp;
    }
}


#endif

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
#ifdef BACKEND_CUDA
    
    Rcpp::stop("Integer not supported for CUDA backend");
    
#else
    std::string my_kernel = as<std::string>(sourceCode_);
    
    std::shared_ptr<viennacl::matrix_range<viennacl::matrix<int> > > vcl_A = getVCLBlockptr<int>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::matrix_range<viennacl::matrix<int> > > vcl_B = getVCLBlockptr<int>(ptrB_, BisVCL, ctx_id);
    std::shared_ptr<viennacl::matrix_range<viennacl::matrix<int> > > vcl_C = getVCLBlockptr<int>(ptrC_, CisVCL, ctx_id);
    
    int M = vcl_A->size2();
    // int N = vcl_B.size1();
    int P = vcl_B->size1();
    int M_internal = vcl_C->internal_size2();
    int P_internal = vcl_C->internal_size1();
    
#ifdef BACKEND_CUDA
    iMatMult<<<max_local_size, max_local_size>>>(M, M_internal,
                                                 P, P_internal,
                                                 viennacl::cuda_arg(*vcl_A),
                                                 viennacl::cuda_arg(*vcl_B),
                                                 viennacl::cuda_arg(*vcl_C));
#else
    
    // get context
    viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
    
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
    my_kernel_mul.global_work_size(0, M_internal);
    my_kernel_mul.global_work_size(1, P_internal);
    
    // set local work sizes
    my_kernel_mul.local_work_size(0, max_local_size);
    my_kernel_mul.local_work_size(1, max_local_size);
    
    // execute kernel
    viennacl::ocl::enqueue(my_kernel_mul(M, M_internal, P, P_internal, *vcl_A, *vcl_B, *vcl_C));
    
#endif
    
    if(!CisVCL){
        // move back to host
        Rcpp::XPtr<dynEigenMat<int> > ptrC(ptrC_);
        
        // copy device data back to CPU
        ptrC->to_host(*vcl_C);
        ptrC->release_device(); 
    }
#endif
}

