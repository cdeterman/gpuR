
#include "gpuR/windows_check.hpp"
#include "gpuR/dynEigenMat.hpp"

template<typename T>
dynEigenMat<T>::dynEigenMat(SEXP A_)
{
    A = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A_);
    nr = A.rows();
    nc = A.cols();
    r_start = 1;
    r_end = nr;
    c_start = 1;
    c_end = nc;
    ptr = A.data();
}

template<typename T>
dynEigenMat<T>::dynEigenMat(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &A_)
{
    A = A_;
    nr = A.rows();
    nc = A.cols();
    r_start = 1;
    r_end = nr;
    c_start = 1;
    c_end = nc;
    ptr = A.data();
}

template<typename T>
dynEigenMat<T>::dynEigenMat(Rcpp::XPtr<dynEigenMat<T> > dynMat)
{
    nr = dynMat->nrow();
    nc = dynMat->ncol();
    r_start = dynMat->row_start();
    r_end = dynMat->row_end();
    c_start = dynMat->col_start();
    c_end = dynMat->col_end();
    ptr = dynMat->getPtr();
}

template<typename T>
dynEigenMat<T>::dynEigenMat(int nr_in, int nc_in)
{
    A = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(nr_in, nc_in);
    nr = nr_in;
    nc = nc_in;
    r_start = 1;
    r_end = nr_in;
    c_start = 1;
    c_end = nc_in;
    ptr = A.data();
}

template<typename T>
dynEigenMat<T>::dynEigenMat(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &A_,
    const int row_start, const int row_end,
    const int col_start, const int col_end)
{
    A = A_;
    nr = A.rows();
    nc = A.cols();
    r_start = row_start-1;
    r_end = row_end-1;
    c_start = col_start-1;
    c_end = col_end-1;
    ptr = A.data();
}

template<typename T>
Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >
dynEigenMat<T>::data() { 
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > temp(ptr, nr, nc);
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > block = temp.block(r_start-1, c_start-1, r_end-r_start + 1, c_end-c_start + 1);
//    std::cout << "internal block" << std::endl;
//    std::cout << block << std::endl;
    return block;
}

template class dynEigenMat<int>;
template class dynEigenMat<float>;
template class dynEigenMat<double>;
