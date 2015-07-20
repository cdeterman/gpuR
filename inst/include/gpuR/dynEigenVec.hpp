#ifndef DYNEIGEN_VEC_HPP
#define DYNEIGEN_VEC_HPP

#include <RcppEigen.h>

template <class T> 
class dynEigenVec {
    private:
        dynEigenVec() { } // private default constructor
        Eigen::Matrix<T, Eigen::Dynamic, 1> A;
        int size;
        
    public:
        dynEigenVec(SEXP A_);
        dynEigenVec(Eigen::Matrix<T, Eigen::Dynamic,1> A_);
        dynEigenVec(int size_in);
        
        T* ptr() { return &A(0); }
        int length() { return size; }
};


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

#endif
