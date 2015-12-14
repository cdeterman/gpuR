
// eigen headers for handling the R input data
#include <RcppEigen.h>

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
void cpp_gpu_eigen(
    SEXP &Am, 
    SEXP &Qm,
    SEXP &eigenvalues,
    bool symmetric,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > > ptrA(Am);
    Rcpp::XPtr<Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > > ptrQ(Qm);
    Rcpp::XPtr<Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > > ptreigenvalues(eigenvalues);
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > eigen_A = *ptrA;
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > &eigen_Q = *ptrQ;
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > &eigen_eigenvalues = *ptreigenvalues;
    
    int M = eigen_A.cols();
    int K = eigen_A.rows();
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_Q(K,M);
    viennacl::vector<T> vcl_eigenvalues(K);
    
    viennacl::copy(eigen_A, vcl_A); 
    viennacl::copy(eigen_Q, vcl_Q); 

    //temp D
    std::vector<T> D(vcl_eigenvalues.size());
    std::vector<T> E(vcl_A.size1());
    
    viennacl::linalg::detail::qr_method(vcl_A, vcl_Q, D, E, symmetric);
    
    viennacl::copy(vcl_Q, eigen_Q);
    std::copy(D.begin(), D.end(), &eigen_eigenvalues(0));
}

template <typename T>
void cpp_vcl_eigen(
    SEXP &Am, 
    SEXP &Qm,
    SEXP &eigenvalues,
    bool symmetric,
    int device_flag)
{    
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<T> > ptrA(Am);
    Rcpp::XPtr<viennacl::matrix<T> > ptrQ(Qm);
    Rcpp::XPtr<viennacl::vector<T> > ptreigenvalues(eigenvalues);
    
    viennacl::matrix<T> vcl_A = *ptrA;
    viennacl::matrix<T> &vcl_Q = *ptrQ;
    viennacl::vector<T> &vcl_eigenvalues = *ptreigenvalues;

    //temp D
    std::vector<T> D(vcl_eigenvalues.size());
    std::vector<T> E(vcl_A.size1());
    
    viennacl::linalg::detail::qr_method(vcl_A, vcl_Q, D, E, symmetric);
    
    // copy D into eigenvalues
    viennacl::copy(D, vcl_eigenvalues);
}


// [[Rcpp::export]]
void
cpp_gpu_eigen(
    SEXP Am, 
    SEXP Qm,
    SEXP eigenvalues,
    const bool symmetric,
    const int type_flag, 
    const int device_flag)
{
    switch(type_flag) {
        case 4:
            cpp_gpu_eigen<int>(Am, Qm, eigenvalues, symmetric, device_flag);
            return;
        case 6:
            cpp_gpu_eigen<float>(Am, Qm, eigenvalues, symmetric, device_flag);
            return;
        case 8:
            cpp_gpu_eigen<double>(Am, Qm, eigenvalues, symmetric, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
void
cpp_vcl_eigen(
    SEXP Am, 
    SEXP Qm,
    SEXP eigenvalues,
    const bool symmetric,
    const int type_flag, 
    const int device_flag)
{
    switch(type_flag) {
        case 4:
            cpp_vcl_eigen<int>(Am, Qm, eigenvalues, symmetric, device_flag);
            return;
        case 6:
            cpp_vcl_eigen<float>(Am, Qm, eigenvalues, symmetric, device_flag);
            return;
        case 8:
            cpp_vcl_eigen<double>(Am, Qm, eigenvalues, symmetric, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}