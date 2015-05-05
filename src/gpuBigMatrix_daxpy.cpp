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
void cpp_gpuBigMatrix_daxpy(SEXP alpha_, SEXP A_, SEXP B_)
{
    if(GPU_HAS_DOUBLE == 0){
        Rcpp::stop("GPU does not support double precision");
    }
    
    const cl_double alpha = as<cl_double>(alpha_);
    static const int incx = 1;
    static const int incy = 1;

    const arma::Mat<double> Am = ConvertBMtoArma<double>(A_);
    arma::Mat<double> Bm = ConvertBMtoArma<double>(B_);
                              
    // total number of elements
    const int N = Am.n_elem;
    
//    Am.print("A Matrix");
//    Bm.print("B Matrix");
    
    // declare OpenCL objects
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem bufA, bufB;
    cl_event event = NULL;
    
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
    ctx = c_createContext(ctx, props, device, err);
    
    queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(ctx);
        stop("clCreateCommandQueue() failed");
    }
    
    /* Setup clblas. */
    err = clblasSetup();
    if (err != CL_SUCCESS) {
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        stop("clblasSetup() failed with " + err);
    }
    
    
//    std::cout << "clblas setup" << std::endl;
    
    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N * sizeof(Am[0]),
                          NULL, &err);
    bufB = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * sizeof(Bm[0]),
                          NULL, &err);
                          
    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
        N * sizeof(Am[0]), &Am[0], 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0,
        N * sizeof(Bm[0]), &Bm[0], 0, NULL, NULL);
    
    /* Call clblas extended function. Perform gemm */
    err = clblasSaxpy(N, alpha, bufA, 0, incx,
                         bufB, 0, incy, 1,
                         &queue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        stop("clblasSaxpy() failed");
    }
    else {
        
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);                                  
        err = clEnqueueReadBuffer(queue, bufB, CL_TRUE, 0,
                                  (N * sizeof(Bm[0])),
                                  &Bm[0], 0, NULL, NULL);
    }
    
    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufA);
    /* Finalize work with clblas. */
    clblasTeardown();
    /* Release OpenCL working objects. */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
}
