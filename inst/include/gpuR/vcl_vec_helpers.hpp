#pragma once
#ifndef VCL_VEC_HELPERS
#define VCL_VEC_HELPERS

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

// ViennaCL headers
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/vector.hpp"

#include <RcppEigen.h>

// convert SEXP Matrix to ViennaCL matrix
template <typename T>
SEXP sexpVecToVCL(SEXP A)
{
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am;
    Am = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A);
    
    //use only GPUs:
    long id = 0;
    viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    
    int M = Am.size();
    
    viennacl::vector<T> *vcl_A = new viennacl::vector<T>(M);
    
    viennacl::copy(Am, *vcl_A); 
    
    Rcpp::XPtr<viennacl::vector<T> > pMat(vcl_A);
    return pMat;
}


// convert XPtr ViennaCL Matrix to Eigen vector
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> VCLtoVecSEXP(SEXP A)
{
    Rcpp::XPtr<viennacl::vector<T> > pA(A);
    
    int M = pA->size();
    std::cout << M << std::endl;
    
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am(M);
    std::cout << Am << std::endl;
    
    viennacl::copy(*pA, Am); 
    
    return Am;
}


// empty ViennaCL Vector
template <typename T>
SEXP emptyVecVCL(int length)
{
    //use only GPUs:
    long id = 0;
    viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    
    viennacl::vector<T> *vcl_A = new viennacl::vector<T>(length);
    *vcl_A = viennacl::zero_vector<T>(length);
    
    Rcpp::XPtr<viennacl::vector<T> > pMat(vcl_A);
    return pMat;
}

#endif