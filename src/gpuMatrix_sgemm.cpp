#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
// clBLAS and OpenCL headers
#include <clBLAS.h>
// armadillo headers for handling the R input data
#include <RcppArmadillo.h>

#include <bigmemory/MatrixAccessor.hpp>

#include "cl_helpers.hpp"

using namespace Rcpp;

// can add more arguments for more control of sgemm call
// e.g. if transpose needed?

//[[Rcpp::export]]
SEXP cpp_gpuMatrix_sgemm(SEXP A_, SEXP B_)
{    
    const clblasOrder order = clblasColumnMajor;
    const cl_float alpha = 1;
    const clblasTranspose transA = clblasNoTrans;

    const arma::Mat<float> Am = as<arma::Mat<float> >(A_);
    const arma::Mat<float> Bm = as<arma::Mat<float> >(B_);
    
    int M = Am.n_cols;
    int K = Am.n_rows;
    int N = Bm.n_rows;
    int P = Bm.n_cols;
    
    int szA = M * N;
    int szB = N * P;
    int szC = K * P;
    
    arma::Mat<float> Cm = arma::Mat<float>(K, P);
    Cm.zeros();

    const std::size_t lda = K;        /* i.e. lda = K */
    const clblasTranspose transB = clblasNoTrans;

    const std::size_t ldb = N;        /* i.e. ldb = N */
    const cl_float beta = 0;
    
    const std::size_t ldc = N;        /* i.e. ldc = N */

    // declare OpenCL objects
    cl_int err;

    // Get available platforms
    std::vector<Platform> platforms;
    getPlatforms(platforms); // cl_helpers.hpp      
    
    // Select the default platform and create a context using this platform and the GPU
    cl_context_properties cps[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platforms[0])(),
        0
    };

    Context context = createContext(CL_DEVICE_TYPE_GPU, cps, err);
            
    // Get a list of devices on this platform
    std::vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    
    // Create a command queue and use the first device
    CommandQueue queue = CommandQueue(context, devices[0], 0, &err);
    
    /* Setup clblas. */
    err = clblasSetup();
    if (err != CL_SUCCESS) {
        stop("clblasSetup() failed with " + err);
    }
        
    // Create memory buffers
    Buffer bufA = Buffer(context, CL_MEM_READ_ONLY, szA * sizeof(float), NULL, &err);
    Buffer bufB = Buffer(context, CL_MEM_READ_ONLY, szB * sizeof(float), NULL, &err);
    Buffer bufC = Buffer(context, CL_MEM_READ_WRITE, szC * sizeof(float), NULL, &err);
    
    // Copy lists A and B to the memory buffers
    queue.enqueueWriteBuffer(bufA, CL_TRUE, 0, szA * sizeof(float), &Am[0]);
    queue.enqueueWriteBuffer(bufB, CL_TRUE, 0, szB * sizeof(float), &Bm[0]);
    
    /* Call clblas extended function. Perform gemm */
    err = clblasSgemm(order, transA, transB, M, N, K,
                         alpha, bufA(), 0, lda,
                         bufB(), 0, ldb, beta,
                         bufC(), 0, ldc,
                         1, &queue(), 0, NULL, 0);
    if (err != CL_SUCCESS) {
        std::cout << err << std::endl;
        stop("clblasSgemmEx() failed");
    }
    else {
        /* Wait for calculations to be finished. */
        err = queue.enqueueReadBuffer(bufC, CL_TRUE, 0, 
                                    szC * sizeof(float), 
                                    &Cm[0]);
    }
    
    /* Release OpenCL memory objects. */
    /* Finalize work with clblas. */
    clblasTeardown();
        
    return wrap(Cm);
}
