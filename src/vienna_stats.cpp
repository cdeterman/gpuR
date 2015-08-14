
// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/eigen_templates.hpp"
#include "gpuR/dynEigen.hpp"

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

// ViennaCL headers
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/sum.hpp"

using namespace Rcpp;

template <typename T>
void cpp_vienna_pmcc(
    MapMat<T> &Am, 
    MapMat<T> &Bm,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    int M = Am.cols();
    int K = Am.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::vector<T> ones = viennacl::scalar_vector<T>(K, 1);
    viennacl::vector<T> vcl_meanVec(M);
    viennacl::matrix<T> vcl_meanMat(K,M);
    
    viennacl::copy(Am, vcl_A); 
    
    // vector of column means
    vcl_meanVec = viennacl::linalg::column_sum(vcl_A);
    vcl_meanVec *= (T)(1)/(T)(K);
    
    // matrix of means
    vcl_meanMat = viennacl::linalg::outer_prod(ones, vcl_meanVec);
    
    viennacl::matrix<T> tmp = vcl_A - vcl_meanMat;
    
    // calculate pearson covariance
    viennacl::matrix<T> vcl_B = viennacl::linalg::prod(trans(tmp), tmp);
    vcl_B *= (T)(1)/(T)(K-1);
    
    viennacl::copy(vcl_B, Bm);
}


//[[Rcpp::export]]
void cpp_vienna_fgpuMatrix_pmcc(
    SEXP ptrA_, 
    SEXP ptrB_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrB(ptrB_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());

    cpp_vienna_pmcc<float>(Am, Bm, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_pmcc(
    SEXP ptrA_, 
    SEXP ptrB_,
    int device_flag)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrB(ptrB_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());

    cpp_vienna_pmcc<double>(Am, Bm, device_flag);
}
