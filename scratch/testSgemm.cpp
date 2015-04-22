#define __CL_ENABLE_EXCEPTIONS

#include <clBLAS.h>

#include <RcppArmadillo.h>

#include "opencl_utils.h"

using namespace Rcpp;

//#define M  4
//#define N  3
//#define K  5


//[[Rcpp::export]]
SEXP cpp_gpu_two_mat2(IntegerMatrix A_, IntegerMatrix B_, 
    IntegerMatrix C_, SEXP sourceCode_, SEXP kernel_function_)
{
    
    static const clblasOrder order = clblasColumnMajor;
    static const cl_float alpha = 1;
    static const clblasTranspose transA = clblasNoTrans;

    // matrix dimensions
//    int M = A_.nrow();
//    int N = B_.ncol();
//    int K = A_.ncol();
    
    int M = A_.ncol();
    int N = B_.nrow();
    int K = A_.nrow();
//    int P = B_.ncol();

    static const std::size_t lda = K;        /* i.e. lda = K */
    static const clblasTranspose transB = clblasNoTrans;

    static const std::size_t ldb = N;        /* i.e. ldb = N */
    static const cl_float beta = 0;
    
    static const std::size_t ldc = N;        /* i.e. ldc = N */

//    static const cl_float A[M*K] = {
//        11, 12, 13, 14, 15,
//        21, 22, 23, 24, 25,
//        31, 32, 33, 34, 35,
//        41, 42, 43, 44, 45,
//    };
    
    
    // use armadillo to fmat
    // then vectorise to fvec by row i.e. vectorise(X, 1);
    
    static const arma::fmat Am = as<arma::fmat>((NumericMatrix)A_);
    static const arma::fmat Bm = as<arma::fmat>((NumericMatrix)B_);
    static arma::fmat Cm = as<arma::fmat>((NumericMatrix)C_);
    
//    std::cout << "Read arma matrices" << std::endl;
    
//    static const arma::fvec A = vectorise(Am);
//    static const arma::fvec B = vectorise(Bm);
//    static arma::fvec C = vectorise(Cm);
    
//    std::cout << "Converted to arma vectors" << std::endl;
    
//    static const std::vector<float> A = as<std::vector<float> >(A_);
//    static const std::vector<float> B = as<std::vector<float> >(B_);
//    static std::vector<float> C = as<std::vector<float> >(C_);
    

//    static const std::vector<float> A { 
////    static const NumericVector A = NumericVector::create(
//        11, 12, 13, 14, 15,
//        21, 22, 23, 24, 25,
//        31, 32, 33, 34, 35,
//        41, 42, 43, 44, 45 
////    );
//    };

//    static const cl_float B[K*N] = {
//        11, 12, 13,
//        21, 22, 23,
//        31, 32, 33,
//        41, 42, 43,
//        51, 52, 53,
//    };

//    static const std::vector<float> B { 
////    static const NumericVector B = NumericVector::create(
//        11, 12, 13,
//        21, 22, 23,
//        31, 32, 33,
//        41, 42, 43,
//        51, 52, 53
////    );
//    };

//    static cl_float C[M*N] = {
//        11, 12, 13,
//        21, 22, 23,
//        31, 32, 33,
//        41, 42, 43,
//    };

//static std::vector<float> C { 
////    static cl_float C[M*N] = {
//        0,0,0,
//        0,0,0,
//        0,0,0,
//        0,0,0
//    };
    
//    static std::vector<int> C { 
//    static NumericVector C = NumericVector::create(
//        11, 12, 13,
//        21, 22, 23,
//        31, 32, 33,
//        41, 42, 43 
//    );
//    static NumericVector C(12);
//    };
        
//    static float result[M*N];
//    static 
    
//    static std::vector<int> result(M*N, 0);
//    static NumericVector result(M*N);
    
    
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem bufA, bufB, bufC;
    cl_event event = NULL;
    int ret = 0;
    
    /* Setup OpenCL environment. */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        stop("clGetPlatformIDs() failed with " + err);
    }
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        stop("clGetDeviceIDs() failed with " + err);
    }
    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        stop("clCreateContext() failed with " + err);
    }
    queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(ctx);
        stop("clCreateCommandQueue() failed with " + err);
    }
    /* Setup clblas. */
    err = clblasSetup();
    if (err != CL_SUCCESS) {
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        stop("clblasSetup() failed with " + err);
    }
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
    /* Call clblas extended function. Perform gemm for the lower right sub-matrices */
    err = clblasSgemm(order, transA, transB, M, N, K,
                         alpha, bufA, 0, lda,
                         bufB, 0, ldb, beta,
                         bufC, 0, ldc,
                         1, &queue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        stop("clblasSgemmEx() failed with " + err);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Fetch results of calculations from GPU memory. */
//        err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
//                                  M * N * sizeof(result[0]),
//                                  &result[0], 0, NULL, NULL);
                                  
        err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
                                  M * N * sizeof(Cm[0]),
                                  &Cm[0], 0, NULL, NULL);
    }
    
    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufC);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufA);
    /* Finalize work with clblas. */
    clblasTeardown();
    /* Release OpenCL working objects. */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    
//    std::vector<int> vec = std::vector<int>(result, result + M*N);
//    std::cout << vec.size() << std::endl;
//    
//    //NumericVector out = as<NumericVector>(vec);
//    return wrap(vec);

    inplace_trans(Cm);
    return wrap(Cm);
//    arma::fmat out(&Cm[0], K, P, false);
//    return wrap(out);
//    return wrap(result);
}
