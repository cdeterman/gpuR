
// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/eigen_templates.hpp"
#include "gpuR/dynEigen.hpp"

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

// ViennaCL headers
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/sum.hpp"

using namespace Rcpp;

template <typename T>
void 
cpp_gpuMatrix_eucl(
    SEXP ptrA_, 
    SEXP ptrD_,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<dynEigen<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<T> > ptrD(ptrD_);
    
    MapMat<T> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<T> Dm(ptrD->ptr(), ptrD->nrow(), ptrD->ncol());
    
    const int M = Am.cols();
    const int K = Am.rows();
    
    // copy to GPU
    viennacl::matrix<T> vcl_A(K,M);    
    viennacl::copy(Am, vcl_A); 
      
    // temp objects
    viennacl::vector<T> row_ones = viennacl::scalar_vector<T>(vcl_A.size1(), 1);
    viennacl::vector<T> vcl_sqrt = viennacl::zero_vector<T>(vcl_A.size2());
    
    // this will definitely need to be updated with the next ViennaCL release
    // currently doesn't support the single scalar operation with
    // element_pow below
    viennacl::matrix<T> twos = viennacl::scalar_matrix<T>(vcl_A.size1(), vcl_A.size2(), 2);
    
    viennacl::matrix<T> square_A = viennacl::linalg::element_pow(vcl_A, twos);
    vcl_sqrt = viennacl::linalg::row_sum(square_A);
    
    viennacl::matrix<T> vcl_D = viennacl::linalg::outer_prod(vcl_sqrt, row_ones);
    
    vcl_D += trans(vcl_D);
    vcl_D -= 2 * (viennacl::linalg::prod(vcl_A, trans(vcl_A)));
    vcl_D = viennacl::linalg::element_sqrt(vcl_D);
    
    for(int i=0; i < vcl_D.size1(); i++){
        vcl_D(i,i) = 0;
    }
    
    viennacl::copy(vcl_D, Dm);
}

template <typename T>
void 
cpp_vclMatrix_eucl(
    SEXP ptrA_, 
    SEXP ptrD_,
    const int device_flag)
{
    //use only GPUs:
    if(device_flag == 0){
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    }
    
    Rcpp::XPtr<viennacl::matrix<T> > ptrA(ptrA_);
    Rcpp::XPtr<viennacl::matrix<T> > ptrD(ptrD_);
    
    viennacl::matrix<T> &vcl_A = *ptrA;
    viennacl::matrix<T> &vcl_D = *ptrD;
    viennacl::vector<T> row_ones = viennacl::scalar_vector<T>(vcl_A.size1(), 1);
    viennacl::vector<T> vcl_sqrt = viennacl::zero_vector<T>(vcl_A.size2());
    
    // this will definitely need to be updated with the next ViennaCL release
    // currently doesn't support the single scalar operation with
    // element_pow below
    viennacl::matrix<T> twos = viennacl::scalar_matrix<T>(vcl_A.size1(), vcl_A.size2(), 2);
    
    viennacl::matrix<T> square_A = viennacl::linalg::element_pow(vcl_A, twos);
    vcl_sqrt = viennacl::linalg::row_sum(square_A);
    
    vcl_D = viennacl::linalg::outer_prod(vcl_sqrt, row_ones);
    
    vcl_D += trans(vcl_D);
    
//    std::cout << vcl_D << std::endl;
    
//    viennacl::matrix<T> temp = 2 * (viennacl::linalg::prod(vcl_A, trans(vcl_A)));
    
//    std::cout << temp << std::endl;
    vcl_D -= 2 * (viennacl::linalg::prod(vcl_A, trans(vcl_A)));
//    vcl_D -= temp;
    vcl_D = viennacl::linalg::element_sqrt(vcl_D);
    
    for(int i=0; i < vcl_D.size1(); i++){
        vcl_D(i,i) = 0;
    }
//    viennacl::diag(vcl_D) = viennacl::zero_vector<T>(vcl_A.size1());
        
}


// [[Rcpp::export]]
void
cpp_vclMatrix_eucl(
    SEXP ptrA, SEXP ptrD,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_vclMatrix_eucl<int>(ptrA, ptrD, device_flag);
            return;
        case 6:
            cpp_vclMatrix_eucl<float>(ptrA, ptrD, device_flag);
            return;
        case 8:
            cpp_vclMatrix_eucl<double>(ptrA, ptrD, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
cpp_gpuMatrix_eucl(
    SEXP ptrA, SEXP ptrD,
    const int device_flag,
    const int type_flag)
{
    
    switch(type_flag) {
        case 4:
            cpp_gpuMatrix_eucl<int>(ptrA, ptrD, device_flag);
            return;
        case 6:
            cpp_gpuMatrix_eucl<float>(ptrA, ptrD, device_flag);
            return;
        case 8:
            cpp_gpuMatrix_eucl<double>(ptrA, ptrD, device_flag);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}
