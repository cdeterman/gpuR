#include <RcppArmadillo.h>
#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>

using namespace Rcpp;

//' @export
// [[Rcpp::export]]
void test_double(SEXP A_, SEXP B_){
    Rcpp::XPtr<BigMatrix> xpA(A_);
    Rcpp::XPtr<BigMatrix> xpB(B_);
        
    static const arma::mat Am = arma::mat( (double*) xpA->matrix(),
                              xpA->nrow(),
                              xpA->ncol(),
                              false);
                              
    static const arma::mat Bm = arma::mat( (double*) xpB->matrix(),
                              xpB->nrow(),
                              xpB->ncol(),
                              false);
    Am.print("A Matrix");
    Bm.print("B Matrix");
}
    

//' @export
// [[Rcpp::export]]
void test_float(SEXP A_, SEXP B_){
    Rcpp::XPtr<BigMatrix> xpA(A_);
    Rcpp::XPtr<BigMatrix> xpB(B_);
        
    static const arma::Mat<float> Am = arma::Mat<float>( (float*) xpA->matrix(),
                              xpA->nrow(),
                              xpA->ncol(),
                              false);
                              
    static const arma::Mat<float> Bm = arma::Mat<float>( (float*) xpB->matrix(),
                              xpB->nrow(),
                              xpB->ncol(),
                              false);
    Am.print("A Matrix");
    Bm.print("B Matrix");
}
