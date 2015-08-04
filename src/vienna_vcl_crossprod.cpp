
#include "gpuR/vcl_helpers.hpp"

// ViennaCL headers
#include "viennacl/linalg/prod.hpp"

using namespace Rcpp;

//[[Rcpp::export]]
void cpp_vclMatrix_dcrossprod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_)
{
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);
    
    *ptrC = viennacl::linalg::prod(trans(*ptrA), *ptrB);
}

//[[Rcpp::export]]
void cpp_vclMatrix_scrossprod(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_)
{    
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);
    
    *ptrC = viennacl::linalg::prod(trans(*ptrA), *ptrB);
}

//[[Rcpp::export]]
void cpp_vclMatrix_dtcrossprod(
    SEXP ptrA_, 
    SEXP ptrB_,
    SEXP ptrC_)
{
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);
    
    *ptrC = viennacl::linalg::prod(*ptrA, trans(*ptrB));
}

//[[Rcpp::export]]
void cpp_vclMatrix_stcrossprod(
    SEXP ptrA_, 
    SEXP ptrB_, 
    SEXP ptrC_)
{   
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);
    
    *ptrC = viennacl::linalg::prod(*ptrA, trans(*ptrB));
}
