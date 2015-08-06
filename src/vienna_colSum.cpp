
// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/eigen_templates.hpp"
#include "gpuR/dynEigen.hpp"
#include "gpuR/dynEigenVec.hpp"

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

// ViennaCL headers
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/sum.hpp"

using namespace Rcpp;

template <typename T>
void cpp_vienna_colsum(
    MapMat<T> &Am, 
    MapVec<T> &colSums)
{    
    //use only GPUs:
    long id = 0;
    viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    
    int M = Am.cols();
    int K = Am.rows();
    int V = colSums.size();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::vector<T> vcl_colSums(V);
    
    viennacl::copy(Am, vcl_A); 
    
    vcl_colSums = viennacl::linalg::column_sum(vcl_A);
    
    viennacl::copy(vcl_colSums, colSums);
}


//[[Rcpp::export]]
void cpp_vienna_fgpuMatrix_colsum(
    SEXP ptrA_, SEXP ptrC_)
{
    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapVec<float> Cm(ptrC->ptr(), ptrC->length());

    cpp_vienna_colsum<float>(Am, Cm);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_colsum(
    SEXP ptrA_, SEXP ptrC_)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapVec<double> Cm(ptrC->ptr(), ptrC->length());

    cpp_vienna_colsum<double>(Am, Cm);
}
