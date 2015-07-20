
// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/eigen_helpers.hpp"

#ifdef VIENNACL_BUILD_INFO
#undef VIENNACL_BUILD_INFO
#endif

#ifdef VIENNACL_DEBUG_ALL
#undef VIENNACL_DEBUG_ALL           //print all of the following
#endif

#ifdef VIENNACL_DEBUG_KERNEL
#undef VIENNACL_DEBUG_KERNEL        //debug any modifications on viennacl::ocl::kernel objects
#endif

#ifdef VIENNACL_DEBUG_COPY
#undef VIENNACL_DEBUG_COPY          //print infos related to setting up/modifying memory objects
#endif

#ifdef VIENNACL_DEBUG_OPENCL
#undef VIENNACL_DEBUG_OPENCL        //display debug info for the OpenCL layer (platform/context/queue creation,
#endif

#ifdef VIENNACL_DEBUG_DEVICE
#undef VIENNACL_DEBUG_DEVICE        //Show device info upon allocation
#endif

#ifdef VIENNACL_DEBUG_CONTEXT
#undef VIENNACL_DEBUG_CONTEXT       //Debug queries to context
#endif

// ViennaCL headers
#include "gpuR/vcl_gemm.hpp"

using namespace Rcpp;

//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_dgemm(SEXP ptrA_, 
                                SEXP ptrB_,
                                SEXP ptrC_)
{
    
    Rcpp::XPtr<dynEigen<double> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<double> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<double> > ptrC(ptrC_);
    
    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<double> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    cpp_arma_vienna_gemm<double>(Am, Bm, Cm);
}

//[[Rcpp::export]]
void cpp_vienna_gpuMatrix_sgemm(SEXP ptrA_, 
                                SEXP ptrB_, 
                                SEXP ptrC_)
{    
    Rcpp::XPtr<dynEigen<float> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<float> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<float> > ptrC(ptrC_);
    
    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<float> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    cpp_arma_vienna_gemm<float>(Am, Bm, Cm);
}
