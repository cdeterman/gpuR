
#include "gpuR/vcl_helpers.hpp"

// ViennaCL headers
#include "viennacl/linalg/prod.hpp"

using namespace Rcpp;

//[[Rcpp::export]]
void cpp_vienna_vclMatrix_dgemm(SEXP ptrA_, 
                                SEXP ptrB_,
                                SEXP ptrC_)
{
    Rcpp::XPtr<viennacl::matrix<double> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<double> > ptrC(ptrC_);
    
//    MapMat<double> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
//    MapMat<double> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
//    MapMat<double> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());

    *ptrC = viennacl::linalg::prod(*ptrA, *ptrB);
}

//[[Rcpp::export]]
void cpp_vienna_vclMatrix_sgemm(SEXP ptrA_, 
                                SEXP ptrB_, 
                                SEXP ptrC_)
{    
    Rcpp::XPtr<viennacl::matrix<float> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrB(ptrB_);
    Rcpp::XPtr<viennacl::matrix<float> > ptrC(ptrC_);
    
//    MapMat<float> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
//    MapMat<float> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
//    MapMat<float> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    *ptrC = viennacl::linalg::prod(*ptrA, *ptrB);
}

////[[Rcpp::export]]
//void cpp_vienna_vclMatrix_igemm(SEXP ptrA_, 
//                                SEXP ptrB_, 
//                                SEXP ptrC_)
//{    
//    Rcpp::XPtr<viennacl::matrix<int> > ptrA(ptrA_);
//    Rcpp::XPtr<viennacl::matrix<int> > ptrB(ptrB_);
//    Rcpp::XPtr<viennacl::matrix<int> > ptrC(ptrC_);
//    
////    MapMat<int> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
////    MapMat<int> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
////    MapMat<int> Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
//    
//    *ptrC = viennacl::linalg::prod(*ptrA, *ptrB);
//}
