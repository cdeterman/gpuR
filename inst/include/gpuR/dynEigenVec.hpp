#pragma once
#ifndef DYNEIGEN_VEC_HPP
#define DYNEIGEN_VEC_HPP

#include <RcppEigen.h>

template <class T> 
class dynEigenVec {
    private:
        dynEigenVec() { } // private default constructor
        Eigen::Matrix<T, Eigen::Dynamic, 1> A;
        int size,begin,last;
        
    public:
        dynEigenVec(SEXP A_);
        dynEigenVec(Eigen::Matrix<T, Eigen::Dynamic,1> A_);
        dynEigenVec(int size_in);
        //dynEigenVec(Eigen::VectorBlock<Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> >, Eigen::Dynamic>);
        dynEigenVec(Eigen::Matrix<T, Eigen::Dynamic,1> A_, const int start, const int end);
        
        T* ptr() { return &A(0); }
        int length() { return size; }
        int start() { return begin; }
        int end() { return last; }
        Eigen::Matrix<T, Eigen::Dynamic, 1> data() { return A; }
};

#endif
