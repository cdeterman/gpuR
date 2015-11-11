
#include "gpuR/windows_check.hpp"
#include "gpuR/dynEigen.hpp"

template<typename T>
dynEigen<T>::dynEigen(SEXP A_)
{
    A = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A_);
    _nr = A.rows();
    _nc = A.cols();
}

template<typename T>
dynEigen<T>::dynEigen(SEXP A_, int nr_in, int nc_in)
{
    A = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A_);
    A.resize(nr_in, nc_in);
    _nr = A.rows();
    _nc = A.cols();
}

template<typename T>
dynEigen<T>::dynEigen(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_)
{
    A = A_;
    _nr = A.rows();
    _nc = A.cols();
}

template<typename T>
dynEigen<T>::dynEigen(int nr_in, int nc_in)
{
    A = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(nr_in,nc_in);
    _nr = nr_in;
    _nc = nc_in;
}

template class dynEigen<int>;
template class dynEigen<float>;
template class dynEigen<double>;
