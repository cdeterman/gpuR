#include "gpuR/windows_check.hpp"

#include <RcppEigen.h>

//#include "gpuR/vcl_helpers.hpp"
#include "gpuR/dynVCLMat.hpp"
#include "gpuR/dynVCLVec.hpp"

using namespace Rcpp;

template <typename T>
int vcl_ncol(SEXP ptrA_)
{
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    viennacl::matrix_range<viennacl::matrix<T> > pA  = ptrA->data();
    return pA.size2();
    
//    Rcpp::XPtr<viennacl::matrix<T> > ptrA(ptrA_);
//    return ptrA->size2();
}

template <typename T>
int vcl_nrow(SEXP ptrA_)
{
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    viennacl::matrix_range<viennacl::matrix<T> > pA  = ptrA->data();
    return pA.size1();
    
//    Rcpp::XPtr<viennacl::matrix<T> > ptrA(ptrA_);
//    return ptrA->size1();
}

template <typename T>
int cpp_vclVector_size(SEXP ptrA_)
{
    Rcpp::XPtr<dynVCLVec<T> > ptrA(ptrA_);
    viennacl::vector_range<viennacl::vector<T> > pA  = ptrA->data();
    return pA.size();
}

// [[Rcpp::export]]
int vcl_dncol(SEXP ptrA)
{
    return vcl_ncol<double>(ptrA);
}

// [[Rcpp::export]]
int vcl_fncol(SEXP ptrA)
{
    return vcl_ncol<float>(ptrA);
}

// [[Rcpp::export]]
int vcl_incol(SEXP ptrA)
{
    return vcl_ncol<int>(ptrA);
}

// [[Rcpp::export]]
int vcl_dnrow(SEXP ptrA)
{
    return vcl_nrow<double>(ptrA);
}

// [[Rcpp::export]]
int vcl_fnrow(SEXP ptrA)
{
    return vcl_nrow<float>(ptrA);
}

// [[Rcpp::export]]
int vcl_inrow(SEXP ptrA)
{
    return vcl_nrow<int>(ptrA);
}

/*** gpuVector size ***/

// [[Rcpp::export]]
SEXP
cpp_vclVector_size(
    SEXP ptrA,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            return wrap(cpp_vclVector_size<int>(ptrA));
        case 6:
            return wrap(cpp_vclVector_size<float>(ptrA));
        case 8:
            return wrap(cpp_vclVector_size<double>(ptrA));
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}
