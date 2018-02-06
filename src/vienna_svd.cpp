
#include "gpuR/windows_check.hpp"

#include "gpuR/dynEigenMat.hpp"
#include "gpuR/dynEigenVec.hpp"
#include "gpuR/dynVCLMat.hpp"
#include "gpuR/dynVCLVec.hpp"

#include "viennacl/matrix.hpp"

#ifndef BACKEND_CUDA
#include "viennacl/linalg/svd.hpp"
#endif

#include <algorithm>

using namespace Rcpp;


template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else
    void
#endif 
cpp_vclMatrix_svd(
    SEXP ptrA_,
    SEXP ptrD_,
    SEXP ptrU_,
    SEXP ptrV_,
    int ctx_id)
{
    
#ifndef BACKEND_CUDA
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > ptrD(ptrD_);
    Rcpp::XPtr<dynVCLMat<T> > ptrU(ptrU_);
    Rcpp::XPtr<dynVCLMat<T> > ptrV(ptrV_);

    // viennacl::matrix_range<viennacl::matrix<T> > A = ptrA->data();
    viennacl::matrix<T> vcl_A = ptrA->matrix();
    
    // viennacl::vector_range<viennacl::vector_base<T> > D  = ptrD->data();
    // viennacl::matrix_range<viennacl::matrix<T> > U = ptrU->data();
    // viennacl::matrix_range<viennacl::matrix<T> > V = ptrV->data();
    // viennacl::vector_base<T> D  = ptrD->vector();
    viennacl::matrix<T> *U = ptrU->getPtr();
    viennacl::matrix<T> *V = ptrV->getPtr();

    // viennacl::matrix<T> U(vcl_A.size1(), vcl_A.size1(), ctx=ctx);
    // viennacl::matrix<T> V(vcl_A.size2(), vcl_A.size2(), ctx = ctx);

    //computes the SVD
    viennacl::linalg::svd(vcl_A, *U, *V);

    // std::cout << "vcl_A" << std::endl;
    // std::cout << vcl_A << std::endl;
    
    viennacl::vector_base<T> D(vcl_A.handle(), std::min(vcl_A.size1(), vcl_A.size2()), 0, vcl_A.internal_size2() + 1);
    
    // D = (vcl_A.handle(), std::min(vcl_A.size1(), vcl_A.size2()), 0, vcl_A.internal_size2() + 1);
    ptrD->setVector(D);
#else
    Rcpp::stop("CUDA backend SVD is not currently implemented :(");
#endif
}


template<typename T>
#ifdef BACKEND_CUDA
typename std::enable_if<std::is_floating_point<T>::value, void>::type
#else
    void
#endif 
cpp_gpuMatrix_svd(
    SEXP ptrA_,
    SEXP ptrD_,
    SEXP ptrU_,
    SEXP ptrV_)
{
    
#ifndef BACKEND_CUDA
    Rcpp::XPtr<dynEigenMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<T> > ptrD(ptrD_);
    Rcpp::XPtr<dynEigenMat<T> > ptrU(ptrU_);
    Rcpp::XPtr<dynEigenMat<T> > ptrV(ptrV_);
    
    // std::cout << "got ptrs" << std::endl;
    
    viennacl::matrix<T> vcl_A = ptrA->device_data();
    viennacl::matrix<T> U = ptrU->device_data();
    viennacl::matrix<T> V = ptrV->device_data();
    
    // std::cout << "got matrices" << std::endl;
    
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > cpuD = ptrD->data();
    
    // std::cout << "got all data" << std::endl;
    
    // std::cout << "A" << std::endl;
    // std::cout << vcl_A << std::endl;
    // 
    // std::cout << "U" << std::endl;
    // std::cout << U << std::endl;
    // 
    // std::cout << "V" << std::endl;
    // std::cout << V << std::endl;
    
    // std::cout << "start svd" << std::endl;
    
    //computes the SVD
    viennacl::linalg::svd(vcl_A, U, V);
    
    // std::cout << "completed svd" << std::endl;
    
    viennacl::vector<T> D = viennacl::diag(vcl_A);
    
    // std::cout << "got diag" << std::endl;
    
    // viennacl::vector_base<T> D(vcl_A.handle(), std::min(vcl_A.size1(), vcl_A.size2()), 0, vcl_A.internal_size2() + 1);
    
    // ptrD->to_host(D);
    viennacl::copy(D, cpuD);
    ptrU->to_host(U);
    ptrV->to_host(V);
#else
    Rcpp::stop("CUDA backend SVD is not currently implemented :(");
#endif
    
}

// [[Rcpp::export]]
void
cpp_vclMatrix_svd(
    SEXP ptrA,
    SEXP ptrD,
    SEXP ptrU,
    SEXP ptrV,
    int type_flag,
    int ctx_id)
{

    switch(type_flag) {
#ifndef BACKEND_CUDA
    case 4:
        cpp_vclMatrix_svd<int>(ptrA, ptrD, ptrU, ptrV, ctx_id);
        return;
#endif
    case 6:
        cpp_vclMatrix_svd<float>(ptrA, ptrD, ptrU, ptrV, ctx_id);
        return;
    case 8:
        cpp_vclMatrix_svd<double>(ptrA, ptrD, ptrU, ptrV, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


// [[Rcpp::export]]
void
cpp_gpuMatrix_svd(
    SEXP ptrA,
    SEXP ptrD,
    SEXP ptrU,
    SEXP ptrV,
    int type_flag)
{
    
    switch(type_flag) {
#ifndef BACKEND_CUDA
    case 4:
        cpp_gpuMatrix_svd<int>(ptrA, ptrD, ptrU, ptrV);
        return;
#endif
    case 6:
        cpp_gpuMatrix_svd<float>(ptrA, ptrD, ptrU, ptrV);
        return;
    case 8:
        cpp_gpuMatrix_svd<double>(ptrA, ptrD, ptrU, ptrV);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


