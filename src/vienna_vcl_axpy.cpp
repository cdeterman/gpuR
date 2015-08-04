
#include "gpuR/vcl_helpers.hpp"

using namespace Rcpp;

//[[Rcpp::export]]
void cpp_vclMatrix_daxpy(SEXP alpha_, 
                        SEXP ptrA_, 
                        SEXP ptrB_)
{
    const double alpha = as<double>(alpha_);
    
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrB(ptrB_);
    
    *ptrB += alpha * (*ptrA);
}

//[[Rcpp::export]]
void cpp_vclMatrix_saxpy(SEXP alpha_, 
                            SEXP ptrA_, 
                            SEXP ptrB_)
{
    const float alpha = as<float>(alpha_);

    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrB(ptrB_);
    
    *ptrB += alpha * (*ptrA);
}
