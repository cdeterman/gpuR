
// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/eigen_helpers.hpp"

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

// ViennaCL headers
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"

using namespace Rcpp;

template <typename T>
inline
void cpp_vienna_crossprod(
    MapMat<T> &Am, 
    MapMat<T> &Bm, 
    MapMat<T> &Cm)
{    
    //use only GPUs:
    long id = 0;
    viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    
    int M = Am.cols();
    int K = Am.rows();
    int N = Bm.rows();
    int P = Bm.cols();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(N,P);
    viennacl::matrix<T> vcl_C(Cm.rows(),Cm.cols());
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_C = viennacl::linalg::prod(trans(vcl_A), vcl_B);
    
    viennacl::copy(vcl_C, Cm);
}

template <typename T>
inline
void cpp_vienna_tcrossprod(
    MapMat<T> &Am, 
    MapMat<T> &Bm, 
    MapMat<T> &Cm)
{    
    //use only GPUs:
    long id = 0;
    viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    
    int M = Am.cols();
    int K = Am.rows();
    int N = Bm.rows();
    int P = Bm.cols();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_B(N,P);
    viennacl::matrix<T> vcl_C(K,N);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_C = viennacl::linalg::prod(vcl_A, trans(vcl_B));
    
    viennacl::copy(vcl_C, Cm);
}
