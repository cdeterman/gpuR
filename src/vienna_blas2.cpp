
#include "gpuR/windows_check.hpp"

// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/dynEigenMat.hpp"
#include "gpuR/dynVCLMat.hpp"
#include "gpuR/dynVCLVec.hpp"

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

// ViennaCL headers
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"

using namespace Rcpp;



/*** vclMatrix Templates ***/

template <typename T>
void cpp_vclMatrix_gemv(
        SEXP ptrA_, 
        SEXP ptrB_,
        SEXP ptrC_)
{
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > ptrB(ptrB_);
    Rcpp::XPtr<dynVCLVec<T> > ptrC(ptrC_);
    
    viennacl::matrix_range<viennacl::matrix<T> > A = ptrA->data();
    viennacl::vector_range<viennacl::vector_base<T> > B = ptrB->data();
    viennacl::vector_range<viennacl::vector_base<T> > C = ptrC->data();
    
    C = viennacl::linalg::prod(A, B);
}

template <typename T>
void cpp_vclMatrix_gevm(
        SEXP ptrA_, 
        SEXP ptrB_,
        SEXP ptrC_)
{
    Rcpp::XPtr<dynVCLVec<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    Rcpp::XPtr<dynVCLVec<T> > ptrC(ptrC_);
    
    viennacl::vector_range<viennacl::vector_base<T> > A = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > B = ptrB->data();
    viennacl::vector_range<viennacl::vector_base<T> > C = ptrC->data();
    
    C = viennacl::linalg::prod(trans(B), A);
}

// [[Rcpp::export]]
void
cpp_vclMatrix_gemv(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    int type_flag)
{
    
    switch(type_flag) {
    case 4:
        cpp_vclMatrix_gemv<int>(ptrA, ptrB, ptrC);
        return;
    case 6:
        cpp_vclMatrix_gemv<float>(ptrA, ptrB, ptrC);
        return;
    case 8:
        cpp_vclMatrix_gemv<double>(ptrA, ptrB, ptrC);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclMatrix_gevm(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    int type_flag)
{
    
    switch(type_flag) {
    case 4:
        cpp_vclMatrix_gevm<int>(ptrA, ptrB, ptrC);
        return;
    case 6:
        cpp_vclMatrix_gevm<float>(ptrA, ptrB, ptrC);
        return;
    case 8:
        cpp_vclMatrix_gevm<double>(ptrA, ptrB, ptrC);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


