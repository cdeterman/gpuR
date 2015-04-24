#ifndef ARMA_HELPERS
#define ARMA_HELPERS

// simple function to get access to big.matrix objects from R
inline 
Rcpp::XPtr<BigMatrix> BigMatrixXPtr(SEXP A){
    // declare as S4 object
    Rcpp::S4 As4(A);
    // pull address slot
    SEXP A_address = As4.slot("address");
    // declare as external pointer
    Rcpp::XPtr<BigMatrix> xpA(A_address);
    return xpA;
 }

// convert either big.matrix or normal matrix 
// to an arma matrix
template <typename T>
arma::Mat<T> ConvertBMtoArma(SEXP A){
    
    Rcpp::XPtr<BigMatrix> xpA = BigMatrixXPtr(A);
        
    arma::Mat<T> Am = arma::Mat<T>( (T*) xpA->matrix(),
              xpA->nrow(),
              xpA->ncol(),
              false);
    return Am;
}

//template <typename T>
//arma::Mat<T> createArma(SEXP A_, bool A_isBM){
//    arma::Mat<T> Am = ( A_isBM ? ConvertBMtoArma<T>(A_) : as<arma::Mat<T> >(A_) );
//    return Am;
//}



//inline
//arma::mat ConvertBMtoArma(SEXP A)
//{
//  Rcpp::XPtr<BigMatrix> xpA = BigMatrixXPtr(A);
//        
//  arma::mat Am = arma::mat( (double*) xpA->matrix(),
//              xpA->nrow(),
//              xpA->ncol(),
//              false);
//  return Am;
//}


//inline
//arma::mat ConvertMatrixToArma(SEXP A, SEXP isBM)
//{
//  if(Rf_asLogical(isBM) == (Rboolean) TRUE)
//      { 
//        Rcpp::XPtr<BigMatrix> xpA = BigMatrixXPtr(A);
//        
//        arma::mat Am = arma::mat( (double*) xpA->matrix(),
//                    xpA->nrow(),
//                    xpA->ncol(),
//                    false);
//        return Am;
//      }else{
//        arma::mat Am = Rcpp::as<arma::mat>(A);
//        return Am;
//      }
//}

#endif