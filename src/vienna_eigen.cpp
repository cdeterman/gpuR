
// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/dynEigenMat.hpp"
#include "gpuR/dynEigenVec.hpp"
#include "gpuR/dynVCLMat.hpp"
#include "gpuR/dynVCLVec.hpp"

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
    
//    Rcpp::XPtr<Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > > ptrA(Am);
//    Rcpp::XPtr<Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > > ptrQ(Qm);
//    Rcpp::XPtr<Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > > ptreigenvalues(eigenvalues);
    Rcpp::XPtr<dynEigenVec<T> > ptreigenvalues(eigenvalues);
    
//    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > eigen_A = *ptrA;
//    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > &eigen_Q = *ptrQ;
//    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > &eigen_eigenvalues = *ptreigenvalues;
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > eigen_eigenvalues = ptreigenvalues->data();
    
    
    XPtr<dynEigenMat<T> > ptrA(Am);
    XPtr<dynEigenMat<T> > ptrQ(Qm);
    
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > refA = ptrA->data();
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > refQ = ptrQ->data();
    
//    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > eigen_A(refA.data(), ptrA->nrow(), ptrA->ncol());
//    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > eigen_Q(refQ.data(), ptrQ->nrow(), ptrQ->ncol());
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::OuterStride<> > eigen_A(
        refA.data(), refA.rows(), refA.cols(),
        Eigen::OuterStride<>(refA.outerStride())
    );
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::OuterStride<> >eigen_Q(
        refQ.data(), refQ.rows(), refQ.cols(),
        Eigen::OuterStride<>(refQ.outerStride())
    );
    
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
    
//    Rcpp::XPtr<viennacl::matrix<T> > ptrA(Am);
//    Rcpp::XPtr<viennacl::matrix<T> > ptrQ(Qm);
    
    
    Rcpp::XPtr<dynVCLMat<T> > ptrA(Am);
    Rcpp::XPtr<dynVCLMat<T> > ptrQ(Qm);
    
//    viennacl::matrix_range<viennacl::matrix<T> > vcl_A = ptrA->data();
//    viennacl::matrix<T> vcl_A = static_cast<viennacl::matrix<T> >(A);
    
//    viennacl::matrix_range<viennacl::matrix<T> > vcl_Q = ptrQ->data();

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
    viennacl::vector_range<viennacl::vector<T> > vcl_eigenvalues  = ptreigenvalues->data();
    
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
