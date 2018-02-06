#include "gpuR/windows_check.hpp"
#include "gpuR/utils.hpp"
#include "gpuR/getVCLptr.hpp"

#define VIENNACL_WITH_EIGEN 1

#include "viennacl/linalg/matrix_operations.hpp"

// using namespace cl;
using namespace Rcpp;

// custom kernels for CUDA backend
#ifdef BACKEND_CUDA

template<typename T>
__global__ void update_kk(T *A, 
                          unsigned int N, unsigned int k)
{
    A[k * N + k] = sqrt(A[k * N + k]);
}

template<typename T>
__global__ void update_k(T *A, 
                         const int upper, unsigned int N, 
                         unsigned int Npad, unsigned int k)
{
    int i = threadIdx.x;
    
    if(i > k && i < N) {
        T Akk = A[k * Npad + k];
        
        if(upper == 0){
            // lower
            A[i * Npad + k] = A[i * Npad + k] / Akk;
            
            // zero out the top too - only if in-place
            A[k * Npad + i] = 0;
        }else if(upper == 1){
            // upper???
            A[k * Npad + i] = A[k * Npad + i] / Akk;
            
            // zero out the top too - only if in-place
            A[i * Npad + k] = 0;
        }else{
            return;
        }
    }
}

template<typename T>
__global__ void update_block(T *A, 
                             const int upper, unsigned int N, 
                             unsigned int Npad, unsigned int k)
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    
    if(i <= k || j <= k) return;
    if(i >= N || j >  i) return;
    
    if(upper == 0){
        // lower
        T Aik = A[i * Npad + k];
        T Ajk = A[j * Npad + k];
        T Aij = A[i * Npad + j];
        
        A[i * Npad + j] = Aij - Aik * Ajk;
        
    }else if(upper ==1 ){
        // upper
        T Aik = A[k * Npad + i];
        T Ajk = A[k * Npad + j];
        T Aij = A[j * Npad + i];
        
        A[j * Npad + i] = Aij - Aik * Ajk;
    }else{
        return;
    }
}

#endif


template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else
void
#endif
cpp_vclMatrix_custom_chol(
    SEXP ptrB_, 
    const bool BisVCL,
    const int upper,
    SEXP sourceCode_,
    unsigned int max_local_size,
    const int ctx_id)
{
    std::string my_kernel = as<std::string>(sourceCode_);
    
    // Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    // Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    // 
    // viennacl::matrix_range<viennacl::matrix<T> > vcl_A  = ptrA->data();
    // viennacl::matrix_range<viennacl::matrix<T> > vcl_B  = ptrB->data();
    
#ifndef BACKEND_CUDA
    viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
#endif
    
    std::shared_ptr<viennacl::matrix_range<viennacl::matrix<T> > > vcl_B = getVCLBlockptr<T>(ptrB_, BisVCL, ctx_id);
    
    unsigned int M = vcl_B->size1();
    // // int N = vcl_B.size1();
    // int P = vcl_B.size2();
    unsigned int M_internal = vcl_B->internal_size2();
    // int P_internal = vcl_B.internal_size1();
    // viennacl::ocl::curr
    
#ifdef BACKEND_CUDA
    for(unsigned int k=0; k < M; k++){
        update_kk<<<1, 1>>>(viennacl::cuda_arg(*vcl_B), M_internal, k);
        update_k<<<1, max_local_size>>>(viennacl::cuda_arg(*vcl_B), upper, M, M_internal, k);
        update_block<<<max_local_size, max_local_size>>>(viennacl::cuda_arg(*vcl_B), upper, M, M_internal, k);
    }
    
#else
    // add kernel to program
    viennacl::ocl::program & my_prog = ctx.add_program(my_kernel, "my_kernel");
    
    // get compiled kernel function
    viennacl::ocl::kernel & update_kk = my_prog.get_kernel("update_kk");
    viennacl::ocl::kernel & update_k = my_prog.get_kernel("update_k");
    viennacl::ocl::kernel & update_block = my_prog.get_kernel("update_block");
    
    // query appropriate max_size and update if different
    check_max_size(ctx, "update_block", max_local_size);
    
    // set global work sizes
    update_kk.global_work_size(0, 1);
    update_k.global_work_size(0, M_internal);
    update_block.global_work_size(0, M_internal);
    update_block.global_work_size(1, M_internal);
    
    // set local work sizes
    update_kk.local_work_size(0, 1);
    update_k.local_work_size(0, max_local_size);
    update_block.local_work_size(0, max_local_size);
    update_block.local_work_size(1, max_local_size);
    
    // execute kernels
    for(unsigned int k=0; k < M; k++){
        viennacl::ocl::enqueue(update_kk(*vcl_B, M_internal, k));
        viennacl::ocl::enqueue(update_k(*vcl_B, upper, M, M_internal, k));
        viennacl::ocl::enqueue(update_block(*vcl_B, upper, M, M_internal, k));
    }
    
#endif
    
    if(!BisVCL){
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        
        // copy device data back to CPU
        ptrB->to_host(*vcl_B.get());
        ptrB->release_device();
    }
}


// [[Rcpp::export]]
void
cpp_vclMatrix_custom_chol(
    SEXP ptrB, 
    const bool BisVCL,
    const int upper,
    SEXP sourceCode,
    int max_local_size,
    const int type_flag,
    const int ctx_id)
{
    switch(type_flag) {
    case 6:
        cpp_vclMatrix_custom_chol<float>(ptrB, BisVCL, upper, sourceCode, max_local_size, ctx_id);
        return;
    case 8:
        cpp_vclMatrix_custom_chol<double>(ptrB, BisVCL, upper, sourceCode, max_local_size, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


