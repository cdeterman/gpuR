
#include "gpuR/windows_check.hpp"
#include "gpuR/dynEigenVec.hpp"

template<typename T>
dynEigenVec<T>::dynEigenVec(SEXP A_)
{
    A = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A_);
    size = A.size();
}

template<typename T>
dynEigenVec<T>::dynEigenVec(Eigen::Matrix<T, Eigen::Dynamic, 1> A_)
{
    A = A_;
    size = A.size();
}

template<typename T>
dynEigenVec<T>::dynEigenVec(int size_in)
{
    A = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(size_in);
    size = size_in;
}

template class dynEigenVec<int>;
template class dynEigenVec<float>;
template class dynEigenVec<double>;
