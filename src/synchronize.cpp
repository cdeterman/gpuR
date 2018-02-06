

#include "gpuR/windows_check.hpp"

// Use OpenCL with ViennaCL
#ifdef BACKEND_CUDA
#define VIENNACL_WITH_CUDA 1
#elif BACKEND_OPENCL
#define VIENNACL_WITH_OPENCL 1
#else
#define VIENNACL_WITH_OPENCL 1
#endif

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

// ViennaCL headers
#include "viennacl/backend/memory.hpp"

#include <RcppEigen.h>


//' @title Synchronize Device Execution
//' @description This pauses execution until the processing is complete
//' on the device (CPU/GPU/etc.).  This is important especially for
//' benchmarking applications.
//' @return NULL
//' @author Charles Determan Jr.
//' @examples \dontrun{
//'     mat <- vclMatrix(rnorm(500^2), ncol = 500, nrow = 500)
//'     system.time(mat %*% mat)
//'     system.time(mat %*% mat; synchronize())
//' }
//' @export
// [[Rcpp::export]]
void synchronize(){
    viennacl::backend::finish();
    return;
}
