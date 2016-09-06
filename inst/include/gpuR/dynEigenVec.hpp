#pragma once
#ifndef DYNEIGEN_VEC_HPP
#define DYNEIGEN_VEC_HPP

#include <RcppEigen.h>


// ViennaCL headers
// #include "viennacl/vector_def.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/vector_proxy.hpp"

template <class T> 
class dynEigenVec {
    private:
        int size,begin,last;
        T* ptr;
        
    public:
        Eigen::Matrix<T, Eigen::Dynamic, 1> A;
        dynEigenVec() { } // private default constructor
        dynEigenVec(SEXP A_);
        dynEigenVec(Eigen::Matrix<T, Eigen::Dynamic,1> &A_);
        dynEigenVec(Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > &A_, int size_);
        dynEigenVec(int size_in);
        dynEigenVec(Eigen::Matrix<T, Eigen::Dynamic,1> &A_, const int start, const int end);
        dynEigenVec(Rcpp::XPtr<dynEigenVec<T> > dynVec);
        
        T* getPtr() { return A.data(); }
        int length() { return size; }
        int start() { return begin; }
        int end() { return last; }
        void setRange(int start, int end){
            begin = start;
            last = end;
        }
        void updateSize(){
            size = last - begin + 1;
        }
        void setVector(Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > &vec){
            A = vec;
        }
        void setPtr(T* ptr_){
            ptr = ptr_;
        }
        //Eigen::Matrix<T, Eigen::Dynamic, 1> data() { return A; }
	    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > data() { 
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > temp(ptr, size, 1);
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > block(&temp(begin-1), last - begin + 1);
            return block;
        }
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > vector() {
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > vec(ptr, size, 1);
//            Eigen::Matrix<T, Eigen::Dynamic, 1>& vec = A;
            return vec;
        }
        // void to_host(viennacl::vector<T> &vclMat);
        
};

#endif
