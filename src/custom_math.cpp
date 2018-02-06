#include "gpuR/windows_check.hpp"

#include "gpuR/utils.hpp"
#include "gpuR/getVCLptr.hpp"

// using namespace cl;
using namespace Rcpp;


// custom kernels for CUDA backend
#ifdef BACKEND_CUDA

template<typename T>
__global__ void MatSign(const T *A, T *B,
                      const unsigned int Mdim, const unsigned int Pdim,
                      const unsigned int MdimPad) { 
    
    int i = threadIdx.x; 
    int j = threadIdx.y; 
    
    // Do the operation
    if((i <= Mdim) && (j <= Pdim)){
        
        // B[i * MdimPad + j] = sign(A[i * MdimPad + j]);
        
        T tmp = A[i * MdimPad + j] < 0 ? -1 : 0;
        B[i * MdimPad + j] = A[i * MdimPad + j] > 0 ? 1 : tmp;
    }
    return;
}

template<typename T>
__global__ void VecSign(const T *A, T *B,
                        const unsigned int n) { 
    
    int i = threadIdx.x;  
    
    // Do the operation
    if(i < n){
        T tmp = A[i] < 0 ? -1 : 0;
        B[i] = A[i] > 0 ? 1 : tmp;
    }
    return;
}

template<typename T>
__global__ void matrix_pmax(
        const T *A, T *B, const T x,
        const unsigned int Mdim, const unsigned Pdim, 
        const unsigned int MdimPad) {
    
    // Get the index of the elements to be processed
    int globalRow = threadIdx.x; // C Row ID
    int globalCol = threadIdx.y; // C Col ID
    
    // Do the operation
    if((globalRow <= Mdim) && (globalCol <= Pdim)){
        
        B[globalRow * MdimPad + globalCol] = max(A[globalRow * MdimPad + globalCol], x);
    }
    return;
}

template<typename T>
__global__ void vector_pmax(
        const T *A, T *B, const T x,
        const unsigned int n) {

    // Get the index of the elements to be processed
    int globalRow = threadIdx.x; // C Row ID

    // Do the operation
    if(globalRow < n){
        B[globalRow] = max(A[globalRow], x);
    }
    return;
}

#endif

template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else 
void
#endif
cpp_vclMatrix_sign(
    SEXP ptrA_,
    const bool AisVCL,
    SEXP ptrB_, 
    const bool BisVCL,
    SEXP sourceCode_,
    unsigned int max_local_size,
    const int ctx_id)
{
    std::string my_kernel = as<std::string>(sourceCode_);
    
    std::shared_ptr<viennacl::matrix_range<viennacl::matrix<T> > > vcl_A = getVCLBlockptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::matrix_range<viennacl::matrix<T> > > vcl_B = getVCLBlockptr<T>(ptrB_, BisVCL, ctx_id);
    
    unsigned int M = vcl_B->size1();
    // // int N = vcl_B.size1();
    unsigned int P = vcl_B->size2();
    unsigned int M_internal = vcl_B->internal_size1();
    unsigned int P_internal = vcl_B->internal_size2();
    
#ifdef BACKEND_CUDA
    MatSign<<<max_local_size, max_local_size>>>(viennacl::cuda_arg(*vcl_A),
                                                viennacl::cuda_arg(*vcl_B),
                                                M, P, M_internal);
#else
    
    viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    // add kernel to program
    viennacl::ocl::program & my_prog = ctx.add_program(my_kernel, "my_kernel");
    
    // get compiled kernel function
    viennacl::ocl::kernel & mat_sign = my_prog.get_kernel("MatSign");
    
    // query appropriate max_size and update if different
    check_max_size(ctx, "MatSign", max_local_size);
    
    // set global work sizes
    mat_sign.global_work_size(0, M_internal);
    mat_sign.global_work_size(1, P_internal);
    
    // set local work sizes
    mat_sign.local_work_size(0, max_local_size);
    mat_sign.local_work_size(1, max_local_size);
    
    // execute kernels
    for(unsigned int k=0; k < M; k++){
        viennacl::ocl::enqueue(mat_sign(*vcl_A, *vcl_B, M, P, M_internal));
    }
    
#endif
    
    if(!BisVCL){
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        
        // copy device data back to CPU
        ptrB->to_host(*vcl_B);
        ptrB->release_device();
    }
}

template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else 
void
#endif
cpp_vclMatrix_pmax(
    SEXP ptrA_,
    const bool AisVCL,
    SEXP ptrB_, 
    const bool BisVCL,
    SEXP value_,
    SEXP sourceCode_,
    unsigned int max_local_size,
    const int ctx_id)
{
    std::string my_kernel = as<std::string>(sourceCode_);
    
    T value = as<T>(value_);
    std::shared_ptr<viennacl::matrix_range<viennacl::matrix<T> > > vcl_A = getVCLBlockptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::matrix_range<viennacl::matrix<T> > > vcl_B = getVCLBlockptr<T>(ptrB_, BisVCL, ctx_id);
    
    unsigned int M = vcl_B->size1();
    unsigned int P = vcl_B->size2();
    unsigned int M_internal = vcl_B->internal_size1();
    unsigned int P_internal = vcl_B->internal_size2();
    
#ifdef BACKEND_CUDA
    matrix_pmax<<<max_local_size, max_local_size>>>(viennacl::cuda_arg(*vcl_A),
                                                    viennacl::cuda_arg(*vcl_B),
                                                    value,
                                                    M, P, M_internal);
#else
    
    viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    // add kernel to program
    viennacl::ocl::program & my_prog = ctx.add_program(my_kernel, "my_kernel");
    
    // get compiled kernel function
    viennacl::ocl::kernel & pmax = my_prog.get_kernel("pmax");
    
    // query appropriate max_size and update if different
    check_max_size(ctx, "pmax", max_local_size);
    
    // set global work sizes
    const unsigned int globalSize1 = roundUp(M_internal, max_local_size);
    const unsigned int globalSize2 = roundUp(P_internal, max_local_size);
    pmax.global_work_size(0, globalSize1);
    pmax.global_work_size(1, globalSize2);
    
    // set local work sizes
    pmax.local_work_size(0, max_local_size);
    pmax.local_work_size(1, max_local_size);
    
    // execute kernels
    for(unsigned int k=0; k < M; k++){
        viennacl::ocl::enqueue(pmax(*vcl_A, *vcl_B, value, M, P, M_internal));
    }
    
#endif
    
    if(!BisVCL){
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
        
        // copy device data back to CPU
        ptrB->to_host(*vcl_B);
        ptrB->release_device();
    }
}


template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else 
void
#endif
cpp_vclVector_sign(
    SEXP ptrA_,
    const bool AisVCL,
    SEXP ptrB_, 
    const bool BisVCL,
    SEXP sourceCode_,
    unsigned int max_local_size,
    const int ctx_id)
{
    std::string my_kernel = as<std::string>(sourceCode_);
    
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_B = getVCLVecptr<T>(ptrB_, BisVCL, ctx_id);
    
    unsigned int M = vcl_B->size();

#ifdef BACKEND_CUDA
    VecSign<<<max_local_size, 1>>>(viennacl::cuda_arg(*vcl_A),
                                   viennacl::cuda_arg(*vcl_B),
                                   M);
#else
    
    viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    // add kernel to program
    viennacl::ocl::program & my_prog = ctx.add_program(my_kernel, "my_kernel");
    
    // get compiled kernel function
    viennacl::ocl::kernel & mat_sign = my_prog.get_kernel("VecSign");
    
    // query appropriate max_size and update if different
    check_max_size(ctx, "VecSign", max_local_size);
    
    // set global work sizes
    int globalSize = roundUp(M, max_local_size);
    mat_sign.global_work_size(0, globalSize);
    
    // set local work sizes
    mat_sign.local_work_size(0, max_local_size);
    
    // execute kernels
    viennacl::ocl::enqueue(mat_sign(*vcl_A, *vcl_B, M));
    
#endif
    
    if(!BisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrB(ptrB_);
        
        // copy device data back to CPU
        ptrB->to_host(*vcl_B);
        ptrB->release_device();
    }
}


template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else 
void
#endif
cpp_vclVector_pmax(
    SEXP ptrA_,
    const bool AisVCL,
    SEXP ptrB_, 
    const bool BisVCL,
    SEXP value_,
    SEXP sourceCode_,
    unsigned int max_local_size,
    const int ctx_id)
{
    std::string my_kernel = as<std::string>(sourceCode_);
    
    T value = as<T>(value_);
    std::shared_ptr<viennacl::vector_base<T> > vcl_A = getVCLVecptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::vector_base<T> > vcl_B = getVCLVecptr<T>(ptrB_, BisVCL, ctx_id);
    
    unsigned int M = vcl_B->size();
    
#ifdef BACKEND_CUDA
    vector_pmax<<<max_local_size, 1>>>(viennacl::cuda_arg(*vcl_A),
                                       viennacl::cuda_arg(*vcl_B),
                                       value,
                                       M);
#else
    
    viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    // add kernel to program
    viennacl::ocl::program & my_prog = ctx.add_program(my_kernel, "my_kernel");
    
    // get compiled kernel function
    viennacl::ocl::kernel & pmax = my_prog.get_kernel("pmax");
    
    // query appropriate max_size and update if different
    check_max_size(ctx, "pmax", max_local_size);
    
    // set global work sizes
    int globalSize = roundUp(M, max_local_size);
    pmax.global_work_size(0, globalSize);
    
    // set local work sizes
    pmax.local_work_size(0, max_local_size);
    
    // execute kernels
    viennacl::ocl::enqueue(pmax(*vcl_A, *vcl_B, value, M));
    
#endif
    
    if(!BisVCL){
        Rcpp::XPtr<dynEigenVec<T> > ptrB(ptrB_);
        
        // copy device data back to CPU
        ptrB->to_host(*vcl_B);
        ptrB->release_device();
    }
}



// [[Rcpp::export]]
void
cpp_vclMatrix_sign(
    SEXP ptrA,
    const bool AisVCL,
    SEXP ptrB, 
    const bool BisVCL,
    SEXP sourceCode,
    int max_local_size,
    const int type_flag,
    const int ctx_id)
{
    switch(type_flag) {
#ifndef BACKEND_CUDA
    case 4:
        cpp_vclMatrix_sign<int>(ptrA, AisVCL, ptrB, BisVCL, sourceCode, max_local_size, ctx_id);
        return;
#endif
    case 6:
        cpp_vclMatrix_sign<float>(ptrA, AisVCL, ptrB, BisVCL, sourceCode, max_local_size, ctx_id);
        return;
    case 8:
        cpp_vclMatrix_sign<double>(ptrA, AisVCL, ptrB, BisVCL, sourceCode, max_local_size, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclVector_sign(
    SEXP ptrA,
    const bool AisVCL,
    SEXP ptrB, 
    const bool BisVCL,
    SEXP sourceCode,
    int max_local_size,
    const int type_flag,
    const int ctx_id)
{
    switch(type_flag) {
#ifndef BACKEND_CUDA
    case 4:
        cpp_vclVector_sign<int>(ptrA, AisVCL, ptrB, BisVCL, sourceCode, max_local_size, ctx_id);
        return;
#endif
    case 6:
        cpp_vclVector_sign<float>(ptrA, AisVCL, ptrB, BisVCL, sourceCode, max_local_size, ctx_id);
        return;
    case 8:
        cpp_vclVector_sign<double>(ptrA, AisVCL, ptrB, BisVCL, sourceCode, max_local_size, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclMatrix_pmax(
    SEXP ptrA,
    const bool AisVCL,
    SEXP ptrB, 
    const bool BisVCL,
    SEXP value,
    SEXP sourceCode,
    int max_local_size,
    const int type_flag,
    const int ctx_id)
{
    switch(type_flag) {
#ifndef BACKEND_CUDA
    case 4:
        cpp_vclMatrix_pmax<int>(ptrA, AisVCL, ptrB, BisVCL, value, sourceCode, max_local_size, ctx_id);
        return;
#endif
    case 6:
        cpp_vclMatrix_pmax<float>(ptrA, AisVCL, ptrB, BisVCL, value, sourceCode, max_local_size, ctx_id);
        return;
    case 8:
        cpp_vclMatrix_pmax<double>(ptrA, AisVCL, ptrB, BisVCL, value, sourceCode, max_local_size, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_vclVector_pmax(
    SEXP ptrA,
    const bool AisVCL,
    SEXP ptrB, 
    const bool BisVCL,
    SEXP value,
    SEXP sourceCode,
    int max_local_size,
    const int type_flag,
    const int ctx_id)
{
    switch(type_flag) {
#ifndef BACKEND_CUDA
    case 4:
        cpp_vclVector_pmax<int>(ptrA, AisVCL, ptrB, BisVCL, value, sourceCode, max_local_size, ctx_id);
        return;
#endif
    case 6:
        cpp_vclVector_pmax<float>(ptrA, AisVCL, ptrB, BisVCL, value, sourceCode, max_local_size, ctx_id);
        return;
    case 8:
        cpp_vclVector_pmax<double>(ptrA, AisVCL, ptrB, BisVCL, value, sourceCode, max_local_size, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}
