#pragma once
#ifndef DYNVCL_VEC_HPP
#define DYNVCL_VEC_HPP

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

// ViennaCL headers
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/vector_proxy.hpp"

#include <RcppEigen.h>

template <class T> 
class dynVCLVec {
    private:
        int size,begin,last;
        viennacl::range r;
        viennacl::vector<T> *ptr;
    
    public:
        viennacl::vector<T> A;
        
        dynVCLVec() { } // private default constructor
        dynVCLVec(viennacl::vector<T> vec, int ctx_id);
        dynVCLVec(SEXP A_, int ctx_id);
//        dynVCLVec(Eigen::Matrix<T, Eigen::Dynamic,1> &A_);
//        dynVCLVec(Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > &A_, int size_);
        dynVCLVec(int size_in, int ctx_id);
//        dynVCLVec(Eigen::Matrix<T, Eigen::Dynamic,1> &A_, const int start, const int end);
        dynVCLVec(Rcpp::XPtr<dynVCLVec<T> > dynVec);
        
        viennacl::vector<T>* getPtr() { return ptr; }
        int length() { return size; }
        int start() { return begin; }
        int end() { return last; }
        void setRange(int start, int end);
        void updateSize();
        void setVector(viennacl::vector_range<viennacl::vector<T> > vec);
        void setVector(viennacl::vector<T> vec);
        void setPtr(viennacl::vector<T>* ptr_);
        viennacl::vector_range<viennacl::vector<T> > data();
        viennacl::vector<T> vector() {
            return A;
        }
        
};

#endif
