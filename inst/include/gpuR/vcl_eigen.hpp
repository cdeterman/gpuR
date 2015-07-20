
// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/eigen_helpers.hpp"

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

// ViennaCL headers
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
//#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/qr-method.hpp"

using namespace Rcpp;

//inline
//void cpp_arma_vienna_eigen(
//    typename MapMat<float>::Type &Am, 
//    typename MapMat<float>::Type &Qm)
//{    
//    //use only GPUs:
//    long id = 0;
//    viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
//    
//    int M = Am.cols();
//    int K = Am.rows();
//    int N = Qm.rows();
//    int P = Qm.cols();
//    
//    std::vector<float> eigenvalues(M);
//    
//    viennacl::matrix<float> vcl_A(K,M);
//    viennacl::matrix<float> vcl_Q(N,P);
//    
//    viennacl::copy(Am, vcl_A); 
//    viennacl::copy(Qm, vcl_Q); 
//    
//    viennacl::linalg::qr_method_sym(vcl_A, vcl_Q, eigenvalues);
//    
//    viennacl::copy(vcl_Q, Qm);
//    
////    for (unsigned int i = 0; i < eigenvalues.size(); i++)
////    {
////        std::cout << std::setprecision(6) << std::fixed << eigenvalues[i] << "\t";
////        std::cout << std::endl;
////    }
//}

template <typename T>
inline
void cpp_vienna_eigen(
    MapMat<T> &Am, 
    MapMat<T> &Qm,
    MapVec<T> &eigenvalues,
    bool symmetric)
{    
    //use only GPUs:
    long id = 0;
    viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    
    int M = Am.cols();
    int K = Am.rows();
    
//    std::vector<T> eigenvalues(M);
    
    viennacl::matrix<T> vcl_A(K,M);
    viennacl::matrix<T> vcl_Q(K,M);
    viennacl::vector<T> vcl_eigenvalues(K);
    
    viennacl::copy(Am, vcl_A); 
    viennacl::copy(Qm, vcl_Q); 
    
//    qr_method(viennacl::matrix<SCALARTYPE> & A,
//                   viennacl::matrix<SCALARTYPE> & Q,
//                   std::vector<SCALARTYPE> & D,
//                   std::vector<SCALARTYPE> & E,
//                   bool is_symmetric = true)

    //temp D
    std::vector<T> D(vcl_eigenvalues.size());
    std::vector<T> E(vcl_A.size1());
    
    viennacl::linalg::detail::qr_method(vcl_A, vcl_Q, D, E, symmetric);
    
    viennacl::copy(vcl_Q, Qm);
    std::copy(D.begin(), D.end(), &eigenvalues(0));
//    viennacl::copy(D, vcl_eigenvalues);
//    viennacl::copy(vcl_eigenvalues, eigenvalues);
    
//    for (unsigned int i = 0; i < eigenvalues.size(); i++)
//    {
//        std::cout << std::setprecision(6) << std::fixed << eigenvalues[i] << "\t";
//        std::cout << std::endl;
//    }
}
