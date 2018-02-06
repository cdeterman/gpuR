
// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/dynEigenMat.hpp"
#include "gpuR/dynEigenVec.hpp"
#include "gpuR/dynVCLMat.hpp"
#include "gpuR/dynVCLVec.hpp"

// ViennaCL headers
#ifndef BACKEND_CUDA
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#endif

#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/qr-method.hpp"

using namespace Rcpp;

template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else
    void
#endif 
    cpp_gpu_eigen(
    SEXP &Am, 
    SEXP &Qm,
    SEXP &eigenvalues,
    bool symmetric)
{    
    Rcpp::XPtr<dynEigenVec<T> > ptreigenvalues(eigenvalues);
    
//    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > eigen_A = *ptrA;
//    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > &eigen_Q = *ptrQ;
//    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > &eigen_eigenvalues = *ptreigenvalues;
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > eigen_eigenvalues = ptreigenvalues->data();

    XPtr<dynEigenMat<T> > ptrA(Am);
    XPtr<dynEigenMat<T> > ptrQ(Qm);
    
    const int K = ptrA->nrow();
    
    viennacl::matrix<T> vcl_A = ptrA->device_data();
    viennacl::matrix<T> vcl_Q = ptrQ->device_data();
    
#ifndef BACKEND_CUDA
    viennacl::context ctx(viennacl::ocl::get_context(ptrA->getContext()));
    viennacl::vector_base<T> vcl_eigenvalues(K, ctx = ctx);
#else
    int cuda_device;
    cudaGetDevice(&cuda_device);
    
    if(ptrA->deviceIndex() != cuda_device){
        cudaSetDevice(ptrA->deviceIndex());    
    }
    
    viennacl::vector_base<T> vcl_eigenvalues(K);
#endif

    //temp D
    std::vector<T> D(vcl_eigenvalues.size());
    std::vector<T> E(vcl_A.size1());
    
    viennacl::linalg::detail::qr_method(vcl_A, vcl_Q, D, E, symmetric);
    
    ptrQ->to_host(vcl_Q);
    
    std::copy(D.begin(), D.end(), &eigen_eigenvalues(0));
}

template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else
    void
#endif 
    cpp_vcl_eigen(
    SEXP &Am, 
    SEXP &Qm,
    SEXP &eigenvalues,
    bool symmetric,
    int ctx_id)
{        
    
    Rcpp::XPtr<dynVCLMat<T> > ptrA(Am);
    Rcpp::XPtr<dynVCLMat<T> > ptrQ(Qm);

    // want copy of A to prevent overwriting original matrix
    viennacl::matrix<T> vcl_A = ptrA->matrix();
    // Q were are overwriting so get pointer
    viennacl::matrix<T> *vcl_Q = ptrQ->getPtr();

//    viennacl::matrix<T> vcl_A = ptrA->matrix();
//    viennacl::matrix<T> vcl_Q = ptrQ->matrix();
    
    // need to find some way to cast without a copy
    
//    viennacl::matrix<T> &vcl_Q = static_cast<viennacl::matrix<T>& >(*Q);
    
//    Rcpp::XPtr<viennacl::vector<T> > ptreigenvalues(eigenvalues);
    
    Rcpp::XPtr<dynVCLVec<T> > ptreigenvalues(eigenvalues);
    viennacl::vector_range<viennacl::vector_base<T> > vcl_eigenvalues  = ptreigenvalues->data();
    
//    viennacl::matrix<T> vcl_A = *ptrA;
//    viennacl::matrix<T> &vcl_Q = *ptrQ;
//    viennacl::vector<T> &vcl_eigenvalues = *ptreigenvalues;

    //temp D
    std::vector<T> D(vcl_eigenvalues.size());
    std::vector<T> E(vcl_A.size1());
    
    viennacl::linalg::detail::qr_method(vcl_A, *vcl_Q, D, E, symmetric);
    
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
    const int type_flag)
{
    switch(type_flag) {
#ifndef BACKEND_CUDA
        case 4:
            cpp_gpu_eigen<int>(Am, Qm, eigenvalues, symmetric);
            return;
#endif
        case 6:
            cpp_gpu_eigen<float>(Am, Qm, eigenvalues, symmetric);
            return;
        case 8:
            cpp_gpu_eigen<double>(Am, Qm, eigenvalues, symmetric);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuMatrix object!");
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
    int ctx_id)
{
    switch(type_flag) {
#ifndef BACKEND_CUDA
        case 4:
            cpp_vcl_eigen<int>(Am, Qm, eigenvalues, symmetric, ctx_id);
            return;
#endif
        case 6:
            cpp_vcl_eigen<float>(Am, Qm, eigenvalues, symmetric, ctx_id);
            return;
        case 8:
            cpp_vcl_eigen<double>(Am, Qm, eigenvalues, symmetric, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}
