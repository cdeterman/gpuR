#define __CL_ENABLE_EXCEPTIONS

// clBLAS and OpenCL headers
#include <clBLAS.h>
// armadillo headers for handling the R input data
#include <RcppArmadillo.h>

#include <bigmemory/MatrixAccessor.hpp>

#include "arma_helpers.hpp"
#include "cl_checks.hpp"
#include "cl_helpers.hpp"

using namespace Rcpp;

// can add more arguments for more control of sgemm call
// e.g. if transpose needed?

//[[Rcpp::export]]
SEXP cpp_gpuMatrix_dgemm(SEXP A_, SEXP B_, SEXP C_)
{
    if(GPU_HAS_DOUBLE == 0){
        Rcpp::stop("GPU does not support double precision");
    }
    
    static const clblasOrder order = clblasColumnMajor;
    static const cl_float alpha = 1;
    static const clblasTranspose transA = clblasNoTrans;
                              
    const arma::Mat<double> Am = as<arma::mat>(A_);
    const arma::Mat<double> Bm = as<arma::mat>(B_);
    arma::Mat<double> Cm = as<arma::mat>(C_);      
    
    int M = Am.n_cols;
    int N = Bm.n_rows;
    int K = Am.n_rows;
    
//    Am.print("A Matrix");
//    Bm.print("B Matrix");

    const std::size_t lda = K;        /* i.e. lda = K */
    static const clblasTranspose transB = clblasNoTrans;

    const std::size_t ldb = N;        /* i.e. ldb = N */
    static const cl_float beta = 0;
    
    const std::size_t ldc = N;        /* i.e. ldc = N */

    // declare OpenCL objects
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem bufA, bufB, bufC;
    cl_event event = NULL;
    
    
//    std::cout << "declared all vars" << std::endl;
    
    /* Setup OpenCL environment. */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        stop("clGetPlatformIDs() failed with " + err);
    }
    
//    std::cout << "found platform" << std::endl;
    
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        stop("clGetDeviceIDs() failed with " + err);
    }
    
//    std::cout << "found device" << std::endl;
    
    props[1] = (cl_context_properties)platform;
    ctx = c_createContext(ctx, props, device, err);
    
//    std::cout << "created context" << std::endl;
    
    queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(ctx);
        stop("clCreateCommandQueue() failed");
    }
    
    
//    std::cout << "opencl setup" << std::endl;
    
    /* Setup clblas. */
    err = clblasSetup();
    if (err != CL_SUCCESS) {
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        stop("clblasSetup() failed with " + err);
    }
    
    
//    std::cout << "clblas setup" << std::endl;
    
    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, M * K * sizeof(Am[0]),
                          NULL, &err);
    bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * N * sizeof(Bm[0]),
                          NULL, &err);
    bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, M * N * sizeof(Cm[0]),
                          NULL, &err);
                          
    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
        M * K * sizeof(Am[0]), &Am[0], 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0,
        K * N * sizeof(Bm[0]), &Bm[0], 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0,
        M * N * sizeof(Cm[0]), &Cm[0], 0, NULL, NULL);
        
    
//    std::cout << "wrote matrices" << std::endl;
    
    /* Call clblas extended function. Perform gemm */
    err = clblasDgemm(order, transA, transB, M, N, K,
                         alpha, bufA, 0, lda,
                         bufB, 0, ldb, beta,
                         bufC, 0, ldc,
                         1, &queue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
//        std::cout << err << std::endl;
        stop("clblasDgemmEx() failed");
    }
    else {
        
//        std::cout << "finished sgemm" << std::endl;
        
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);                                  
        err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
                                  M * N * sizeof(Cm[0]),
                                  &Cm[0], 0, NULL, NULL);
    }
    
    
//    std::cout << "read output" << std::endl;
    
    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufC);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufA);
    /* Finalize work with clblas. */
    clblasTeardown();
    /* Release OpenCL working objects. */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return wrap(Cm);
}
