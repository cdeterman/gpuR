#include <iostream>

#include "gpuR/windows_check.hpp"
#include "gpuR/dynVCLMat.hpp"


template<typename T>
dynVCLMat<T>::dynVCLMat(viennacl::matrix<T> mat, int ctx_id){

    viennacl::context ctx;

    // explicitly pull context for thread safe forking
    ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));

    // must explicity switch context to make sure the same
    // it appears when class initialized the A is set to current context (may not be desired)
    A.switch_memory_context(ctx);
    A = mat;

    nr = A.size1();
    nc = A.size2();
    ptr = &A;
    shptr = std::make_shared<viennacl::matrix<T> >(A);
    // shptr.reset(ptr);
    viennacl::range temp_rr(0, nr);
    viennacl::range temp_cr(0, nc);
    row_r = temp_rr;
    col_r = temp_cr;
}

template<typename T>
dynVCLMat<T>::dynVCLMat(SEXP A_, int ctx_id)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Am;
    Am = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A_);
    
    int K = Am.rows();
    int M = Am.cols();
    viennacl::context ctx;
    
    // explicitly pull context for thread safe forking
    ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));

    A.switch_memory_context(ctx);
    A = viennacl::matrix<T>(K,M, ctx);
      
    viennacl::copy(Am, A); 
    
    nr = K;
    nc = M;
    ptr = &A;
    
    shptr = std::make_shared<viennacl::matrix<T> >(A);
    // shptr.reset(ptr);
    viennacl::range temp_rr(0, nr);
    viennacl::range temp_cr(0, nc);
    row_r = temp_rr;
    col_r = temp_cr;
}

template<typename T>
dynVCLMat<T>::dynVCLMat(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Am, 
    int nr_in, int nc_in,
    int ctx_id
    )
{    
    viennacl::context ctx;
    
    // explicitly pull context for thread safe forking
    ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
 
    A.switch_memory_context(ctx);
    A = viennacl::matrix<T>(nr_in, nc_in, ctx);
    viennacl::copy(Am, A); 
    
    nr = nr_in;
    nc = nc_in;
    ptr = &A;
    shptr = std::make_shared<viennacl::matrix<T> >(A);
    // shptr.reset(ptr);
    viennacl::range temp_rr(0, nr);
    viennacl::range temp_cr(0, nc);
    row_r = temp_rr;
    col_r = temp_cr;
}

template<typename T>
dynVCLMat<T>::dynVCLMat(int nr_in, int nc_in, int ctx_id)
{    
    viennacl::context ctx;
    
    // explicitly pull context for thread safe forking
    ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));

    A.switch_memory_context(ctx);
    A = viennacl::zero_matrix<T>(nr_in, nc_in, ctx);
       
    nr = nr_in;
    nc = nc_in;
    ptr = &A;
    shptr = std::make_shared<viennacl::matrix<T> >(A);
    // shptr.reset(ptr);
    viennacl::range temp_rr(0, nr);
    viennacl::range temp_cr(0, nc);
    row_r = temp_rr;
    col_r = temp_cr;
}

template<typename T>
dynVCLMat<T>::dynVCLMat(int nr_in, int nc_in, T scalar, int ctx_id)
{
    viennacl::context ctx;
    
    // explicitly pull context for thread safe forking
    ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
    
    A.switch_memory_context(ctx);
    A = viennacl::scalar_matrix<T>(nr_in, nc_in, scalar, ctx);
    
    nr = nr_in;
    nc = nc_in;
    ptr = &A;
    shptr = std::make_shared<viennacl::matrix<T> >(A);
    // shptr.reset(ptr);
    viennacl::range temp_rr(0, nr);
    viennacl::range temp_cr(0, nc);
    row_r = temp_rr;
    col_r = temp_cr;
}


// template<typename T>
// dynVCLMat<T>::dynVCLMat(Rcpp::XPtr<dynVCLMat<T> > dynMat)
// {
//     nr = dynMat->nrow();
//     nc = dynMat->ncol();
//     row_r = dynMat->row_range();
//     col_r = dynMat->col_range();
//     ptr = dynMat->getPtr();
// }

template<typename T>
void 
dynVCLMat<T>::setRange(
    int row_start, int row_end,
    int col_start, int col_end)
{
    viennacl::range temp_rr(row_start, row_end);
    viennacl::range temp_cr(col_start, col_end);
    row_r = temp_rr;
    col_r = temp_cr;
}


// template<typename T>
// void
// dynVCLMat<T>::createMatrix(int nr_in, int nc_in, int ctx_id){
// 
//     // std::cout << "creating matrix" << std::endl;
// 
//     viennacl::context ctx;
// 
//     // explicitly pull context for thread safe forking
//     ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
// 
//     // std::cout << "pulled context" << std::endl;
//     // std::cout << ctx_id << std::endl;
// 
//     A = viennacl::matrix<T>(nr_in, nc_in, ctx=ctx);
// 
//     // std::cout << "assigned new matrix" << std::endl;
// 
//     ptr = &A;
// }

template<typename T>
void
dynVCLMat<T>::setMatrix(viennacl::matrix<T> mat){
    A = mat;
    ptr = &A;
    shptr = std::make_shared<viennacl::matrix<T> >(A);
    // shptr.reset(ptr);
}

// template<typename T>
// void
// dynVCLMat<T>::setMatrix(viennacl::matrix_range<viennacl::matrix<T> > mat){
//     A = mat;
//     ptr = &A;
// }

template<typename T>
void 
dynVCLMat<T>::setPtr(viennacl::matrix<T>* ptr_){
    ptr = ptr_;
    shptr = std::make_shared<viennacl::matrix<T> >(*ptr);
    // shptr.reset(ptr_);
}

template<typename T>
void
dynVCLMat<T>::setSharedPtr(std::shared_ptr<viennacl::matrix<T> > shptr_){
    shptr = shptr_;
}

template<typename T>
viennacl::matrix_range<viennacl::matrix<T> >
dynVCLMat<T>::data() { 
    viennacl::matrix_range<viennacl::matrix<T> > m_sub(*shptr.get(), row_r, col_r);
    return m_sub;
}

template<typename T>
viennacl::matrix<T>* 
dynVCLMat<T>::getPtr() { 
    return shptr.get(); 
}

template<typename T>
std::shared_ptr<viennacl::matrix<T> >
dynVCLMat<T>::sharedPtr() {
    return shptr;
}

template class dynVCLMat<int>;
template class dynVCLMat<float>;
template class dynVCLMat<double>;
