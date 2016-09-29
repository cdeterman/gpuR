
#include "gpuR/windows_check.hpp"

// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/dynEigenMat.hpp"
#include "gpuR/dynVCLMat.hpp"
#include "gpuR/getVCLptr.hpp"

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

#include "viennacl/matrix.hpp"
#include "viennacl/linalg/lu.hpp"

#include <memory>


template <typename T>
void cpp_gpuMatrix_solve(
        SEXP ptrA_,
        SEXP ptrB_,
        const bool AisVCL,
        const bool BisVCL,
        const int ctx_id)
{
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    viennacl::matrix<T> *vcl_A;
    viennacl::matrix<T> *vcl_B;

    vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);

    // solution of a full system right into the load vector vcl_rhs:
    viennacl::linalg::lu_factorize(*vcl_A);
    viennacl::linalg::lu_substitute(*vcl_A, *vcl_B);

    if(!BisVCL){
        Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);

        // copy device data back to CPU
        ptrB->to_host(*vcl_B);
        ptrB->release_device();
    }
}


// [[Rcpp::export]]
void
cpp_gpuMatrix_solve(
    SEXP ptrA,
    SEXP ptrB,
    bool AisVCL,
    bool BisVCL,
    const int type_flag,
    const int ctx_id)
{

    switch(type_flag) {
        case 6:
            cpp_gpuMatrix_solve<float>(ptrA, ptrB, AisVCL, BisVCL, ctx_id);
            return;
        case 8:
            cpp_gpuMatrix_solve<double>(ptrA, ptrB, AisVCL, BisVCL, ctx_id);
            return;
        default:
            throw Rcpp::exception("unknown type detected for gpuR matrix object!");
    }
}

