
#include "gpuR/windows_check.hpp"
#include "gpuR/dynEigenVec.hpp"

template<typename T>
dynEigenVec<T>::dynEigenVec(SEXP A_)
{
    A = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A_);
    size = A.size();
    begin = 1;
    last = size;
    ptr = A.data();
}

template<typename T>
dynEigenVec<T>::dynEigenVec(Eigen::Matrix<T, Eigen::Dynamic, 1> &A_)
{
    A = A_;
    size = A.size();
    begin = 1;
    last = size;
    ptr = A.data();
}

// template<typename T>
// dynEigenVec<T>::dynEigenVec(Rcpp::XPtr<dynEigenVec<T> > dynVec)
// {
//     size = dynVec->length();
//     begin = dynVec->start();
//     last = dynVec->end();
//     ptr = dynVec->getPtr();
// }
// 
// template<typename T>
// dynEigenVec<T>::dynEigenVec(Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > &A_, int size_){
//     A = A_;
//     size = A.size();
//     begin = 1;
//     last = size;
//     ptr = A.data();
// }

template<typename T>
dynEigenVec<T>::dynEigenVec(int size_in)
{
    A = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(size_in);
    size = size_in;
    begin = 1;
    last = size;
    ptr = A.data();
}

// template<typename T>
// void
// dynEigenVec<T>::to_host(viennacl::vector<T> &vclMat) {
//     // Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > temp(ptr, size, 1);
//     // Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > block(&temp(begin-1), last - begin + 1);
//     viennacl::copy(vclMat, A);
// }

// template<typename T>
// dynEigenVec<T>::dynEigenVec(
//     Eigen::Matrix<T, Eigen::Dynamic, 1> &A_,
//     const int start,
//     const int end)
// {
//     A = A_;
//     size = A.size();
//     begin = start - 1;
//     last = end - 1;
//     ptr = A.data();
// }

template class dynEigenVec<int>;
template class dynEigenVec<float>;
template class dynEigenVec<double>;
