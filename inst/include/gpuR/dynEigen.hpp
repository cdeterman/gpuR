#pragma once
#ifndef DYNEIGEN_HPP
#define DYNEIGEN_HPP

#include <RcppEigen.h>

template <class T> 
class dynEigen {
    private:
        dynEigen() { } // private default constructor
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A;
        int _nr, _nc;
        
    public:
        dynEigen(SEXP A_);
        dynEigen(SEXP A_, int nr_in, int nc_in);
        dynEigen(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_);
        dynEigen(int nr_in, int nc_in);
        
        T* ptr() { return &A(0); }
        int nrow() { return _nr; }
        int ncol() { return _nc; }
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> data() { return A; }
};

#endif
