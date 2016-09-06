// 
// #include "gpuR/windows_check.hpp"
// 
// // eigen headers for handling the R input data
// #include <RcppEigen.h>
// 
// #include "gpuR/dynEigenMat.hpp"
// #include "gpuR/dynVCLMat.hpp"
// 
// // Use OpenCL with ViennaCL
// #define VIENNACL_WITH_OPENCL 1
// 
// // Use ViennaCL algorithms on Eigen objects
// #define VIENNACL_WITH_EIGEN 1
// 
// #include "viennacl/matrix.hpp"
// #include "viennacl/linalg/prod.hpp"
// #include "viennacl/linalg/qr.hpp"
// 
// using namespace Rcpp;
// 
// template <typename T>
// void
// cpp_vclMatrix_qr(
//     SEXP ptrA_,
//     SEXP ptrQ_,
//     SEXP ptrR_,
//     int ctx_id)
// {    
//     
//     viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
//     
//     Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
//     Rcpp::XPtr<dynVCLMat<T> > ptrQ(ptrQ_);
//     Rcpp::XPtr<dynVCLMat<T> > ptrR(ptrR_);
//     
//     // viennacl::matrix_range<viennacl::matrix<T> > A = ptrA->data();
//     viennacl::matrix<T> vcl_A = ptrA->matrix();
//     
//     // viennacl::matrix<T> Q(vcl_A.size1(), vcl_A.size1(), ctx=ctx);
//     // viennacl::matrix<T> R(vcl_A.size1(), vcl_A.size2(), ctx = ctx);
//     // viennacl::matrix<T> QR(vcl_A.size1(), vcl_A.size2(), ctx = ctx);
//     
//     viennacl::matrix<T> *Q = ptrQ->getPtr();
//     viennacl::matrix<T> *R = ptrR->getPtr();
//     
//     //computes the QR factorization
//     std::vector<T> betas = viennacl::linalg::inplace_qr(vcl_A); 
//     
//     viennacl::linalg::recoverQ(vcl_A, betas, *Q, *R);
//     // QR = viennacl::linalg::prod(Q, R);
//     
//     // std::cout << "Q" << std::endl;
//     // std::cout << Q << std::endl;
//     // 
//     // std::cout << "R" << std::endl;
//     // std::cout << R << std::endl;
//     // 
//     std::cout << "betas" << std::endl;
//     Rcout << NumericVector(betas.begin(), betas.end()) << std::endl;
// }
// 
// 
// // [[Rcpp::export]]
// void
// cpp_vclMatrix_qr(
//     SEXP ptrA,
//     SEXP ptrQ,
//     SEXP ptrR,
//     int type_flag,
//     int ctx_id)
// {
//     
//     switch(type_flag) {
//     case 4:
//         cpp_vclMatrix_qr<int>(ptrA, ptrQ, ptrR, ctx_id);
//         return;
//     case 6:
//         cpp_vclMatrix_qr<float>(ptrA, ptrQ, ptrR, ctx_id);
//         return;
//     case 8:
//         cpp_vclMatrix_qr<double>(ptrA, ptrQ, ptrR, ctx_id);
//         return;
//     default:
//         throw Rcpp::exception("unknown type detected for vclMatrix object!");
//     }
// }
// 
