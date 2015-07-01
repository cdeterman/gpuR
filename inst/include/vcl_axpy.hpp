
// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "eigen_helpers.hpp"

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

//#ifdef VIENNACL_DEBUG_ALL
//#undef VIENNACL_DEBUG_ALL
//#endif
//
//#ifdef VIENNACL_DEBUG_DEVICE
//#undef VIENNACL_DEBUG_DEVICE
//#endif

// ViennaCL headers
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/matrix.hpp"

using namespace Rcpp;

template <typename T>
inline
void cpp_arma_vienna_axpy(
    T const alpha, 
    MapMat<T> &Am, 
    MapMat<T> &Bm)
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
    viennacl::matrix<T> vcl_C(K,P);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Bm, vcl_B); 
    
    vcl_B += alpha * vcl_A;

    viennacl::copy(vcl_B, Bm);
}
