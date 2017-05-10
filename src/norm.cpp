
#include "gpuR/windows_check.hpp"

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
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/sum.hpp"
#include "viennacl/linalg/maxmin.hpp"
#include "viennacl/linalg/norm_frobenius.hpp"
#include "viennacl/linalg/svd.hpp"

#include <unordered_map>

using namespace Rcpp;

typedef enum {One,Infinity,Frobenius,Maximum_Modulus,Spectral} NORM_METHOD;

const std::unordered_map<std::string, NORM_METHOD> norm_methods {
    {"O", One},
    {"o", One},
    {"1", One},
    {"I", Infinity},
    {"i", Infinity},
    {"F", Frobenius},
    {"f", Frobenius},
    {"M", Maximum_Modulus},
    {"m", Maximum_Modulus},
    {"2", Spectral}
};

template <typename T>
T cpp_vclMatrix_norm_one(SEXP ptrA_)
{
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    
    viennacl::matrix_range<viennacl::matrix<T> > vcl_A = ptrA->data();
    
    T result = viennacl::linalg::max(viennacl::linalg::column_sum(viennacl::linalg::element_fabs(vcl_A)));
    
    return result;
}

template <typename T>
T cpp_vclMatrix_norm_inf(SEXP ptrA_)
{
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    
    viennacl::matrix_range<viennacl::matrix<T> > vcl_A = ptrA->data();
    
    T result = viennacl::linalg::max(viennacl::linalg::row_sum(viennacl::linalg::element_fabs(vcl_A)));
    
    return result;
}

template <typename T>
T cpp_vclMatrix_norm_frobenius(SEXP ptrA_)
{
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    
    viennacl::matrix_range<viennacl::matrix<T> > vcl_A = ptrA->data();
    
    T result = viennacl::linalg::norm_frobenius(vcl_A);
    
    return result;
}

template <typename T>
T cpp_vclMatrix_norm_max_mod(SEXP ptrA_)
{
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    
    viennacl::matrix_range<viennacl::matrix<T> > vcl_A = ptrA->data();
    
    viennacl::vector_base<T> tmp(vcl_A.handle(), vcl_A.internal_size(), 0, 1);
    
    T result = viennacl::linalg::max(viennacl::linalg::element_fabs(tmp));
    
    return result;
}


template <typename T>
T cpp_vclMatrix_norm_2(SEXP ptrA_)
{
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    viennacl::matrix<T> vcl_A = ptrA->matrix();
    viennacl::context ctx = ptrA->getContext();
    
    if(vcl_A.size1() != vcl_A.size2()){
        stop("only square matrices currently supported");
    }
        
    viennacl::matrix<T> U = viennacl::zero_matrix<T>(vcl_A.size1(), vcl_A.size1(), ctx);
    viennacl::matrix<T> V = viennacl::zero_matrix<T>(vcl_A.size2(), vcl_A.size2(), ctx);
        
    //computes the SVD
    viennacl::linalg::svd(vcl_A, U, V);
    
    // get singular values
    viennacl::vector_base<T> D(vcl_A.handle(), std::min(vcl_A.size1(), vcl_A.size2()), 0, vcl_A.internal_size2() + 1);
    
    T result = viennacl::linalg::max(D);
    
    return result;
}

// [[Rcpp::export]]
SEXP
cpp_vclMatrix_norm(
    SEXP ptrA,
    std::string method,
    const int type_flag)
{
    
    auto nm = norm_methods.find(method);
    if(nm == norm_methods.end()){
        stop("method not supported");
    }
    NORM_METHOD n_method = nm->second;
    
    switch(n_method) {
    case One: {
        switch(type_flag) {
        case 4:
            return wrap(cpp_vclMatrix_norm_one<int>(ptrA));
        case 6:
            return wrap(cpp_vclMatrix_norm_one<float>(ptrA));
        case 8:
            return wrap(cpp_vclMatrix_norm_one<double>(ptrA));
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
        }
    }
    case Infinity: {
        switch(type_flag) {
        case 4:
            return wrap(cpp_vclMatrix_norm_inf<int>(ptrA));
        case 6:
            return wrap(cpp_vclMatrix_norm_inf<float>(ptrA));
        case 8:
            return wrap(cpp_vclMatrix_norm_inf<double>(ptrA));
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
        }
    }   
    case Frobenius: {
        switch(type_flag) {
        case 4:
            return wrap(cpp_vclMatrix_norm_frobenius<int>(ptrA));
        case 6:
            return wrap(cpp_vclMatrix_norm_frobenius<float>(ptrA));
        case 8:
            return wrap(cpp_vclMatrix_norm_frobenius<double>(ptrA));
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
        }
    }
    case Maximum_Modulus: {
        switch(type_flag) {
        case 4:
            return wrap(cpp_vclMatrix_norm_max_mod<int>(ptrA));
        case 6:
            return wrap(cpp_vclMatrix_norm_max_mod<float>(ptrA));
        case 8:
            return wrap(cpp_vclMatrix_norm_max_mod<double>(ptrA));
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
        }
    }
    case Spectral: {
        switch(type_flag) {
        case 4:
            return wrap(cpp_vclMatrix_norm_2<int>(ptrA));
        case 6:
            return wrap(cpp_vclMatrix_norm_2<float>(ptrA));
        case 8:
            return wrap(cpp_vclMatrix_norm_2<double>(ptrA));
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
        }
    }
    default:
        throw Rcpp::exception("unknown norm method");
    }
    
}

