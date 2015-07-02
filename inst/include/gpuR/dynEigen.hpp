#ifndef DYNEIGEN_HPP
#define DYNEIGEN_HPP

#include <RcppEigen.h>

template <class T> 
class dynEigen {
    private:
        dynEigen() { } // private default constructor
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A;
        int nr, nc;
        
    public:
        dynEigen(SEXP A_);
        dynEigen(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_);
        dynEigen(int nr_in, int nc_in);
        
        T* ptr() { return &A(0); }
        int nrow() { return nr; }
        int ncol() { return nc; }
};


template<typename T>
dynEigen<T>::dynEigen(SEXP A_)
{
    A = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A_);
    nr = A.rows();
    nc = A.cols();
}

template<typename T>
dynEigen<T>::dynEigen(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_)
{
    A = A_;
    nr = A.rows();
    nc = A.rows();
}

template<typename T>
dynEigen<T>::dynEigen(int nr_in, int nc_in)
{
    A = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(nr_in,nc_in);
    nr = nr_in;
    nc = nc_in;
}

#endif
