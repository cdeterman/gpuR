
#include "gpuR/windows_check.hpp"

#include "gpuR/dynEigenMat.hpp"
#include "gpuR/dynVCLMat.hpp"
#include "gpuR/getVCLptr.hpp"

#include "viennacl/matrix.hpp"
#include "viennacl/linalg/lu.hpp"

#include <memory>

template <typename T>
T cpp_gpuMatrix_det(
        SEXP ptrA_,
        const bool AisVCL,
        const int ctx_id)
{
    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);

    Eigen::Matrix<T, Eigen::Dynamic, 1> A = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(vcl_A->size1());
    
    
    // solution of a full system right into the load vector vcl_rhs:
    viennacl::linalg::lu_factorize(*vcl_A);
    
    viennacl::vector<T> vA = viennacl::diag(*vcl_A);
    
    viennacl::fast_copy(vA.begin(), vA.end(), &(A[0]));
    
    T det = A.prod();
    
    if(!AisVCL){
        Rcpp::XPtr<dynEigenMat<T> > ptrA(ptrA_);
        
        // release GPU memory
        ptrA->release_device();
    }
    
    return det;
}


template <typename T>
void cpp_gpuMatrix_solve(
        SEXP ptrA_,
        SEXP ptrB_,
        const bool AisVCL,
        const bool BisVCL,
        const int ctx_id)
{

    std::shared_ptr<viennacl::matrix<T> > vcl_A = getVCLptr<T>(ptrA_, AisVCL, ctx_id);
    std::shared_ptr<viennacl::matrix<T> > vcl_B = getVCLptr<T>(ptrB_, BisVCL, ctx_id);

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
SEXP
cpp_gpuMatrix_det(
    SEXP ptrA,
    bool AisVCL,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
    case 6:
        return Rcpp::wrap(cpp_gpuMatrix_det<float>(ptrA, AisVCL, ctx_id));
    case 8:
        return Rcpp::wrap(cpp_gpuMatrix_det<double>(ptrA, AisVCL, ctx_id));
    default:
        throw Rcpp::exception("unknown type detected for gpuR matrix object!");
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

