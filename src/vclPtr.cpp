#include <RcppEigen.h>

#include "gpuR/vcl_helpers.hpp"

#include "viennacl/vector.hpp"

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::MatrixXi;
using Eigen::VectorXd;
using Eigen::VectorXf;
using Eigen::VectorXi;

using namespace Rcpp;


// convert SEXP Vector to ViennaCL matrix
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


// convert XPtr ViennaCL Vector to Eigen vector
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> VCLtoVecSEXP(SEXP A)
{
    Rcpp::XPtr<viennacl::vector<T> > pA(A);
    
    int M = pA->size();
    
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am(M);
    
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


/*** matrix imports ***/

// [[Rcpp::export]]
SEXP matrixToIntVCL(SEXP data)
{
    SEXP pMat = sexpToVCL<int>(data);
    return(pMat);
}

// [[Rcpp::export]]
SEXP matrixToFloatVCL(SEXP data)
{
    SEXP pMat = sexpToVCL<float>(data);
    return(pMat);
}

// [[Rcpp::export]]
SEXP matrixToDoubleVCL(SEXP data)
{
    SEXP pMat = sexpToVCL<double>(data);
    return(pMat);
}


/*** Matrix exports ***/

// [[Rcpp::export]]
SEXP dVCLtoSEXP(SEXP ptrA)
{
    MatrixXd A = VCLtoSEXP<double>(ptrA);
    return wrap(A);
}


// [[Rcpp::export]]
SEXP fVCLtoSEXP(SEXP ptrA)
{
    MatrixXf A = VCLtoSEXP<float>(ptrA);
    return wrap(A);
}


// [[Rcpp::export]]
SEXP iVCLtoSEXP(SEXP ptrA)
{
    MatrixXi A = VCLtoSEXP<int>(ptrA);
    return wrap(A);
}

/*** Empty matrix initializers ***/

// [[Rcpp::export]]
SEXP emptyIntVCL(int nr, int nc)
{
    SEXP pMat = emptyVCL<int>(nr,nc);
    return(pMat);
}


// [[Rcpp::export]]
SEXP emptyFloatVCL(int nr, int nc)
{
    SEXP pMat = emptyVCL<float>(nr,nc);
    return(pMat);
}


// [[Rcpp::export]]
SEXP emptyDoubleVCL(int nr, int nc)
{
    SEXP pMat = emptyVCL<double>(nr,nc);
    return(pMat);
}

/*** vector imports ***/

// [[Rcpp::export]]
SEXP vectorToIntVCL(SEXP data)
{
    SEXP pMat = sexpVecToVCL<int>(data);
    return(pMat);
}

// [[Rcpp::export]]
SEXP vectorToFloatVCL(SEXP data)
{
    SEXP pMat = sexpVecToVCL<float>(data);
    return(pMat);
}

// [[Rcpp::export]]
SEXP vectorToDoubleVCL(SEXP data)
{
    SEXP pMat = sexpVecToVCL<double>(data);
    return(pMat);
}


/*** Vector exports ***/

// [[Rcpp::export]]
SEXP dVCLtoVecSEXP(SEXP ptrA)
{
    VectorXd A = VCLtoVecSEXP<double>(ptrA);
    return wrap(A);
}


// [[Rcpp::export]]
SEXP fVCLtoVecSEXP(SEXP ptrA)
{
    VectorXf A = VCLtoVecSEXP<float>(ptrA);
    return wrap(A);
}


// [[Rcpp::export]]
SEXP iVCLtoVecSEXP(SEXP ptrA)
{
    VectorXi A = VCLtoVecSEXP<int>(ptrA);
    return wrap(A);
}

/*** Empty vector initializers ***/

// [[Rcpp::export]]
SEXP emptyVecIntVCL(int length)
{
    SEXP pMat = emptyVecVCL<int>(length);
    return(pMat);
}


// [[Rcpp::export]]
SEXP emptyVecFloatVCL(int length)
{
    SEXP pMat = emptyVecVCL<float>(length);
    return(pMat);
}


// [[Rcpp::export]]
SEXP emptyVecDoubleVCL(int length)
{
    SEXP pMat = emptyVecVCL<double>(length);
    return(pMat);
}

