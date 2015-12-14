#pragma once
#ifndef VCL_HELPERS
#define VCL_HELPERS

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

// ViennaCL headers
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"

#include <RcppEigen.h>

// convert SEXP Matrix to ViennaCL matrix
template <typename T>
SEXP sexpToVCL(SEXP A)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Am;
    Am = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A);
    
    //use only GPUs:
    long id = 0;
    viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    
    int K = Am.rows();
    int M = Am.cols();
    
    viennacl::matrix<T> *vcl_A = new viennacl::matrix<T>(K,M);
    
    viennacl::copy(Am, *vcl_A); 
    
    Rcpp::XPtr<viennacl::matrix<T> > pMat(vcl_A);
    return pMat;
}


// convert XPtr ViennaCL Matrix to Eigen matrix
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> 
VCLtoSEXP(SEXP A)
{
    Rcpp::XPtr<viennacl::matrix<T> > pA(A);
    
    int nr = pA->size1();
    int nc = pA->size2();
    
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Am(nr, nc);
    
    viennacl::copy(*pA, Am); 
    
    return Am;
}


// empty ViennaCL matrix
template <typename T>
SEXP cpp_zero_vclMatrix(int nr, int nc)
{
    //use only GPUs:
    long id = 0;
    viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    
    viennacl::matrix<T> *vcl_A = new viennacl::matrix<T>(nr,nc);
    *vcl_A = viennacl::zero_matrix<T>(nr,nc);
    
    Rcpp::XPtr<viennacl::matrix<T> > pMat(vcl_A);
    return pMat;
}



#endif
