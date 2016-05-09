

#include "gpuR/windows_check.hpp"
#include "gpuR/dynVCLMat.hpp"

template<typename T>
dynVCLMat<T>::dynVCLMat(SEXP A_, int context_index)
{

    // remove context_id argument
    
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Am;
    Am = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A_);
    
    int K = Am.rows();
    int M = Am.cols();
    
    A = viennacl::matrix<T>(K,M);
      
    viennacl::copy(Am, A); 
    
    nr = K;
    nc = M;
    ptr = &A;
    viennacl::range temp_rr(0, nr);
    viennacl::range temp_cr(0, nc);
    row_r = temp_rr;
    col_r = temp_cr;
}

template<typename T>
dynVCLMat<T>::dynVCLMat(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Am, 
    int nr_in, int nc_in,
    int context_index
    )
{    

    // remove context_id argument

    A = viennacl::matrix<T>(nr_in, nc_in);
    viennacl::copy(Am, A); 
    
    nr = nr_in;
    nc = nc_in;
    ptr = &A;
    viennacl::range temp_rr(0, nr);
    viennacl::range temp_cr(0, nc);
    row_r = temp_rr;
    col_r = temp_cr;
}

template<typename T>
dynVCLMat<T>::dynVCLMat(int nr_in, int nc_in)
{    
    A = viennacl::zero_matrix<T>(nr_in, nc_in);
       
    nr = nr_in;
    nc = nc_in;
    ptr = &A;
    viennacl::range temp_rr(0, nr);
    viennacl::range temp_cr(0, nc);
    row_r = temp_rr;
    col_r = temp_cr;
}

template<typename T>
dynVCLMat<T>::dynVCLMat(int nr_in, int nc_in, int context_index)
{
    A = viennacl::zero_matrix<T>(nr_in, nc_in);
    
    nr = nr_in;
    nc = nc_in;
    ptr = &A;
    viennacl::range temp_rr(0, nr);
    viennacl::range temp_cr(0, nc);
    row_r = temp_rr;
    col_r = temp_cr;
}

template<typename T>
dynVCLMat<T>::dynVCLMat(int nr_in, int nc_in, T scalar, int context_index)
{
    A = viennacl::scalar_matrix<T>(nr_in, nc_in, scalar);
    
    nr = nr_in;
    nc = nc_in;
    ptr = &A;
    viennacl::range temp_rr(0, nr);
    viennacl::range temp_cr(0, nc);
    row_r = temp_rr;
    col_r = temp_cr;
}


template<typename T>
dynVCLMat<T>::dynVCLMat(Rcpp::XPtr<dynVCLMat<T> > dynMat)
{
    nr = dynMat->nrow();
    nc = dynMat->ncol();
    row_r = dynMat->row_range();
    col_r = dynMat->col_range();
    ptr = dynMat->getPtr();
}

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

template<typename T>
void
dynVCLMat<T>::setMatrix(viennacl::matrix<T> mat){
    A = mat;
    ptr = &A;
}

template<typename T>
void
dynVCLMat<T>::setMatrix(viennacl::matrix_range<viennacl::matrix<T> > mat){
    A = mat;
    ptr = &A;
}

template<typename T>
void 
dynVCLMat<T>::setPtr(viennacl::matrix<T>* ptr_){
    ptr = ptr_;
}

template<typename T>
viennacl::matrix_range<viennacl::matrix<T> >
dynVCLMat<T>::data() { 
    viennacl::matrix_range<viennacl::matrix<T> > m_sub(*ptr, row_r, col_r);
    return m_sub;
}

template class dynVCLMat<int>;
template class dynVCLMat<float>;
template class dynVCLMat<double>;
