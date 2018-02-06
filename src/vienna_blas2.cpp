
#include "gpuR/windows_check.hpp"

// eigen headers for handling the R input data
#include <RcppEigen.h>

// #include "gpuR/dynEigenMat.hpp"
// #include "gpuR/dynEigenVec.hpp"
#include "gpuR/dynVCLMat.hpp"
#include "gpuR/dynVCLVec.hpp"

// ViennaCL headers
#ifndef BACKEND_CUDA
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#endif
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"

using namespace Rcpp;



/*** vclMatrix Templates ***/

template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else
    void
#endif
    cpp_vclMatrix_gemv(
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

template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else
    void
#endif
    cpp_vclMatrix_gevm(
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


template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else
    void
#endif
cpp_vclMatVec_crossprod(
    SEXP ptrA_, 
    const bool AisVec,
    SEXP ptrB_,
    const bool BisVec,
    SEXP ptrC_)
{
    if(AisVec){
        Rcpp::XPtr<dynVCLVec<T> > ptrA(ptrA_);
        Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
        Rcpp::XPtr<dynVCLVec<T> > ptrC(ptrC_);
        
        viennacl::vector_range<viennacl::vector_base<T> > A = ptrA->data();
        viennacl::matrix_range<viennacl::matrix<T> > B = ptrB->data();
        viennacl::vector_range<viennacl::vector_base<T> > C = ptrC->data();
        
        C = viennacl::linalg::prod(trans(B), A);
    }else{
        if(BisVec){
            Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
            Rcpp::XPtr<dynVCLVec<T> > ptrB(ptrB_);
            Rcpp::XPtr<dynVCLVec<T> > ptrC(ptrC_);
            
            viennacl::matrix_range<viennacl::matrix<T> > A = ptrA->data();
            viennacl::vector_range<viennacl::vector_base<T> > B = ptrB->data();
            viennacl::vector_range<viennacl::vector_base<T> > C = ptrC->data();
            
            C = viennacl::linalg::prod(trans(A), B);
        }else{
            throw Rcpp::exception("one of the objects must be a vector");
        }
    }
}

template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else
    void
#endif
cpp_vclMatVec_tcrossprod(
    SEXP ptrA_, 
    const bool AisVec,
    SEXP ptrB_,
    const bool BisVec,
    SEXP ptrC_,
    const bool CisVec)
{
    if(AisVec){
        Rcpp::XPtr<dynVCLVec<T> > ptrA(ptrA_);
        Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
        
        viennacl::vector_range<viennacl::vector_base<T> > A = ptrA->data();
        viennacl::matrix_range<viennacl::matrix<T> > B = ptrB->data();
        
        if(CisVec){
            Rcpp::XPtr<dynVCLVec<T> > ptrC(ptrC_);
            viennacl::vector_range<viennacl::vector_base<T> > C = ptrC->data();
            C = viennacl::linalg::prod(B, A);
        }else{
            Rcpp::XPtr<dynVCLMat<T> > ptrC(ptrC_);
            viennacl::matrix_range<viennacl::matrix<T> > C = ptrC->data();

            viennacl::vector<T> B_vec = viennacl::column(B, 0);

            // this won't work for ranges!!!!
            // viennacl::matrix_base<float> dummy(A.handle(),
            //                                    vcl_A.size(), 0, 1, vcl_A.size(),   //row layout
            //                                    1, 0, 1, 1,   //column layout
            //                                    true); // row-major
            // C = viennacl::linalg::prod(B, trans(dummy));

            C = viennacl::linalg::outer_prod(A, B_vec);
        }
    }else{
        if(BisVec){
            Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
            Rcpp::XPtr<dynVCLVec<T> > ptrB(ptrB_);
            
            viennacl::matrix_range<viennacl::matrix<T> > A = ptrA->data();
            viennacl::vector_range<viennacl::vector_base<T> > B = ptrB->data();
            
            viennacl::vector<T> A_vec = viennacl::column(A, 0);
            
            if(CisVec){
                Rcpp::XPtr<dynVCLVec<T> > ptrC(ptrC_);
                viennacl::vector_range<viennacl::vector_base<T> > C = ptrC->data();
                C = viennacl::linalg::prod(trans(A), B);
            }else{
                Rcpp::XPtr<dynVCLMat<T> > ptrC(ptrC_);
                viennacl::matrix_range<viennacl::matrix<T> > C = ptrC->data();
                
                // viennacl::matrix_base<float> dummy(vcl_B.handle(),
                //                                    vcl_B.size(), 0, 1, vcl_B.size(),   //row layout
                //                                    1, 0, 1, 1,   //column layout
                //                                    true); // row-major
                
                C = viennacl::linalg::outer_prod(A_vec, B);
            }
        }else{
            throw Rcpp::exception("one of the objects must be a vector");
        }
    }
}

template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else
    void
#endif
    cpp_vclMatVec_axpy(
        SEXP alpha_,
        SEXP ptrA_, 
        const bool AisVec,
        SEXP ptrB_,
        const bool BisVec,
        const int ctx_id)
{
    
    const T alpha = as<T>(alpha_);
    
    if(AisVec){
        Rcpp::XPtr<dynVCLVec<T> > ptrA(ptrA_);
        Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
        
        viennacl::vector_range<viennacl::vector_base<T> > A = ptrA->data();
        viennacl::matrix_range<viennacl::matrix<T> > vcl_B = ptrB->data();
        
        viennacl::matrix_base<T> vcl_A = viennacl::matrix_base<T>(A.handle(),
                                                                  vcl_B.size2(), 0, 1, vcl_B.size2(),   //row layout
                                                                  vcl_B.size1(), 0, 1, vcl_B.size1(),   //column layout
                                                                  true); // row-major
        vcl_B += alpha * trans(vcl_A);
    }else{
        if(BisVec){
            Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
            Rcpp::XPtr<dynVCLVec<T> > ptrB(ptrB_);
            
            viennacl::matrix_range<viennacl::matrix<T> > vcl_A = ptrA->data();
            viennacl::vector_range<viennacl::vector_base<T> > B = ptrB->data();
            
            viennacl::matrix_base<T> vcl_B = viennacl::matrix_base<T>(B.handle(),
                                                                      vcl_A.size2(), 0, 1, vcl_A.size2(),   //row layout
                                                                      vcl_A.size1(), 0, 1, vcl_A.size1(),   //column layout
                                                                      true); // row-major
            
            vcl_B += alpha * trans(vcl_A);
        }else{
            throw Rcpp::exception("one of the objects must be a vector");
        }
    }
}

// template <typename T>
// void cpp_gpuMatVec_axpy(
//         SEXP alpha_,
//         SEXP ptrA_, 
//         const bool AisVec,
//         SEXP ptrB_,
//         const bool BisVec,
//         const int ctx_id)
// {
//     
//     viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
//     
//     const T alpha = as<T>(alpha_);
//     
//     if(AisVec){
//         Rcpp::XPtr<dynEigenVec<T> > ptrA(ptrA_);
//         Rcpp::XPtr<dynEigenMat<T> > ptrB(ptrB_);
//         
//         viennacl::vector<T> A = ptrA->data();
//         viennacl::matrix<T> vcl_B = ptrB->data();
//         
//         viennacl::matrix_base<T> vcl_A = viennacl::matrix_base<T>(A.handle(),
//                                                                   vcl_B.size2(), 0, 1, vcl_B.size2(),   //row layout
//                                                                   vcl_B.size1(), 0, 1, vcl_B.size1(),   //column layout
//                                                                   true); // row-major
//         vcl_B += alpha * trans(vcl_A);
//         ptrB->to_host();
//         
//     }else{
//         if(BisVec){
//             Rcpp::XPtr<dynEigenMat<T> > ptrA(ptrA_);
//             Rcpp::XPtr<dynEigenVec<T> > ptrB(ptrB_);
//             
//             viennacl::matrix<T> vcl_A = ptrA->data();
//             viennacl::vector<T> B = ptrB->data();
//             
//             viennacl::matrix_base<T> vcl_B = viennacl::matrix_base<T>(B.handle(),
//                                                                       vcl_A.size2(), 0, 1, vcl_A.size2(),   //row layout
//                                                                       vcl_A.size1(), 0, 1, vcl_A.size1(),   //column layout
//                                                                       true); // row-major
//             
//             vcl_B += alpha * trans(vcl_A);
//             ptrB->to_host(vcl_B);
//             
//         }else{
//             throw Rcpp::exception("one of the objects must be a vector");
//         }
//     }
// }

// [[Rcpp::export]]
void
cpp_vclMatrix_gemv(
    SEXP ptrA, SEXP ptrB, SEXP ptrC,
    int type_flag)
{
    
    switch(type_flag) {
#ifndef BACKEND_CUDA
    case 4:
        cpp_vclMatrix_gemv<int>(ptrA, ptrB, ptrC);
        return;
#endif
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
#ifndef BACKEND_CUDA
    case 4:
        cpp_vclMatrix_gevm<int>(ptrA, ptrB, ptrC);
        return;
#endif
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

// [[Rcpp::export]]
void
cpp_vclMatVec_crossprod(
    SEXP ptrA, 
    const bool AisVec, 
    SEXP ptrB, 
    const bool BisVec, 
    SEXP ptrC,
    const int type_flag)
{
    
    switch(type_flag) {
#ifndef BACKEND_CUDA
    case 4:
        cpp_vclMatVec_crossprod<int>(ptrA, AisVec, ptrB, BisVec, ptrC);
        return;
#endif
    case 6:
        cpp_vclMatVec_crossprod<float>(ptrA, AisVec, ptrB, BisVec, ptrC);
        return;
    case 8:
        cpp_vclMatVec_crossprod<double>(ptrA, AisVec, ptrB, BisVec, ptrC);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclMatVec_tcrossprod(
    SEXP ptrA, 
    const bool AisVec, 
    SEXP ptrB, 
    const bool BisVec, 
    SEXP ptrC,
    const bool CisVec,
    const int type_flag)
{
    
    switch(type_flag) {
#ifndef BACKEND_CUDA
    case 4:
        cpp_vclMatVec_tcrossprod<int>(ptrA, AisVec, ptrB, BisVec, ptrC, CisVec);
        return;
#endif
    case 6:
        cpp_vclMatVec_tcrossprod<float>(ptrA, AisVec, ptrB, BisVec, ptrC, CisVec);
        return;
    case 8:
        cpp_vclMatVec_tcrossprod<double>(ptrA, AisVec, ptrB, BisVec, ptrC, CisVec);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


// [[Rcpp::export]]
void
cpp_vclMatVec_axpy(
    SEXP alpha,
    SEXP ptrA, 
    const bool AisVec,
    SEXP ptrB, 
    const bool BisVec,
    const int type_flag,
    const int ctx_id)
{
    
    switch(type_flag) {
#ifndef BACKEND_CUDA
    case 4:
        cpp_vclMatVec_axpy<int>(alpha, ptrA, AisVec, ptrB, BisVec, ctx_id);
        return;
#endif
    case 6:
        cpp_vclMatVec_axpy<float>(alpha, ptrA, AisVec, ptrB, BisVec, ctx_id);
        return;
    case 8:
        cpp_vclMatVec_axpy<double>(alpha, ptrA, AisVec, ptrB, BisVec, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}
