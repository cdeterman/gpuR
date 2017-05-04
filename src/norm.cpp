
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
        stop("Infinity not yet implemented");
    }   
    case Frobenius: {
        stop("Frobenius not yet implemented");
    }
    case Maximum_Modulus: {
        stop("Maximum modulus not yet implemented");
    }
    case Spectral: {
        stop("Spectral not yet implemented");
    }
    default:
        throw Rcpp::exception("unknown norm method");
    }
    
}

