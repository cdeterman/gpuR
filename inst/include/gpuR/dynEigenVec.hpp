#pragma once
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

#endif
