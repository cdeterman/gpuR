// Armadillo headers (disable BLAS and LAPACK to avoid linking issues)
#define ARMA_DONT_USE_BLAS
#define ARMA_DONT_USE_LAPACK

// armadillo headers for handling the R input data
#include <RcppArmadillo.h>

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Armadillo objects
#define VIENNACL_WITH_ARMADILLO 1

// ViennaCL headers
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"

using namespace Rcpp;

template <typename T>
inline
arma::Mat<T> cpp_arma_vienna_gemm(const arma::Mat<T> &Am, const arma::Mat<T> &Bm)
{    
    //use only GPUs:
    long id = 0;
    viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    
    int M = Am.n_cols;
    int K = Am.n_rows;
    int N = Bm.n_rows;
    int P = Bm.n_cols;
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(N,P);
    viennacl::matrix<T> vcl_C(K,P);
    
    arma::Mat<T> Cm = arma::Mat<T>(K, P);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_C = viennacl::linalg::prod(vcl_A, vcl_B);
    
    viennacl::copy(vcl_C, Cm);
    
    return Cm;
}