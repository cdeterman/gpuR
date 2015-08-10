
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
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/qr-method.hpp"

using namespace Rcpp;

template <typename T>
void cpp_vienna_eigen(
    MapMat<T> &Am, 
    MapMat<T> &Qm,
    MapVec<T> &eigenvalues,
    bool symmetric,
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
    viennacl::matrix<T> vcl_Q(K,M);
    viennacl::vector<T> vcl_eigenvalues(K);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Qm, vcl_Q); 

    //temp D
    std::vector<T> D(vcl_eigenvalues.size());
    std::vector<T> E(vcl_A.size1());
    
    viennacl::linalg::detail::qr_method(vcl_A, vcl_Q, D, E, symmetric);
    
    viennacl::copy(vcl_Q, Qm);
    std::copy(D.begin(), D.end(), &eigenvalues(0));
}


//[[Rcpp::export]]
void cpp_vienna_fgpuMatrix_eigen(
    SEXP ptrA_, SEXP ptrB_, SEXP ptrC_, 
    bool symmetric, int device_flag)
{
    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigenVec<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapVec<float> Cm(ptrC->ptr(), ptrC->length());

    cpp_vienna_eigen<float>(Am, Bm, Cm, symmetric, device_flag);
}

//[[Rcpp::export]]
void cpp_vienna_dgpuMatrix_eigen(
    SEXP ptrA_, SEXP ptrB_, SEXP ptrC_,
    bool symmetric, int device_flag)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigenVec<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapVec<double> Cm(ptrC->ptr(), ptrC->length());

    cpp_vienna_eigen<double>(Am, Bm, Cm, symmetric, device_flag);
}
