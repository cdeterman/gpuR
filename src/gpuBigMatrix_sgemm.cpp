#define __CL_ENABLE_EXCEPTIONS
// C++ OpenCL API
#include <CL/cl.hpp>
// clBLAS and OpenCL headers
#include <clBLAS.h>
// armadillo headers for handling the R input data
#include <RcppArmadillo.h>

#include <bigmemory/MatrixAccessor.hpp>

#include "arma_helpers.hpp"
#include "cl_helpers.hpp"

using namespace Rcpp;

// can add more arguments for more control of sgemm call
// e.g. if transpose needed?

//[[Rcpp::export]]
void cpp_gpuBigMatrix_sgemm(SEXP A_, SEXP B_, SEXP C_)
{
    
    static const clblasOrder order = clblasColumnMajor;
    static const cl_float alpha = 1;
    static const clblasTranspose transA = clblasNoTrans;

    const arma::Mat<float> Am = ConvertBMtoArma<float>(A_);
    const arma::Mat<float> Bm = ConvertBMtoArma<float>(B_);
    arma::Mat<float> Cm = ConvertBMtoArma<float>(C_);
                              
    const int M = Am.n_cols;
    const int N = Bm.n_rows;
    const int K = Am.n_rows;
    
    const int szA = Am.n_elem;
    const int szB = Bm.n_elem;
    const int szC = Cm.n_elem;

    const std::size_t lda = K;        /* i.e. lda = K */
    static const clblasTranspose transB = clblasNoTrans;

    const std::size_t ldb = N;        /* i.e. ldb = N */
    static const cl_float beta = 0;
    
    const std::size_t ldc = N;        /* i.e. ldc = N */

    // declare OpenCL objects
    cl_int err;
    
    /* Setup OpenCL environment. */
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
    
    /* Prepare OpenCL memory objects and place matrices inside them. */
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
        stop("clblasSgemmEx() failed");
    }
    else {
        /* Wait for calculations to be finished. */ 
        err = queue.enqueueReadBuffer(bufC, CL_TRUE, 0, 
                                    M * N * sizeof(float), 
                                    &Cm[0]);
    }
    
    /* Finalize work with clblas. */
    clblasTeardown();
}
