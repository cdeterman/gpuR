#pragma once
#ifndef DYNVCL_MAT_HPP
#define DYNVCL_MAT_HPP

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

// ViennaCL headers
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"

#include <RcppEigen.h>

template <class T> 
class dynVCLMat {
    private:
        int nr, nc;
        viennacl::range row_r;
        viennacl::range col_r;
        viennacl::matrix<T> *ptr;
    
    public:
        viennacl::matrix<T> A;
        
        dynVCLMat() { } // private default constructor
        dynVCLMat(SEXP A_, int device_flag);
        dynVCLMat(
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Am,
            int nr_in, int nc_in,
            int device_flag
            );
        dynVCLMat(int nr_in, int nc_in, int device_flag);
        dynVCLMat(int nr_in, int nc_in, T scalar, int device_flag);
        dynVCLMat(Rcpp::XPtr<dynVCLMat<T> > dynMat);
        
        viennacl::matrix<T>* getPtr() { return ptr; }
        int nrow() { return nr; }
        int ncol() { return nc; }
        viennacl::range row_range() { return row_r; }
        viennacl::range col_range() { return col_r; }
        
        void setRange(
            int row_start, int row_end,
            int col_start, int col_end
            );
        void setMatrix(viennacl::matrix_range<viennacl::matrix<T> > mat){
            A = mat;
            ptr = &A;
        }
        void setDims(int nr_in, int nc_in){
            nr = nr_in;
            nc = nc_in;
        }
        void setPtr(viennacl::matrix<T>* ptr_);
        viennacl::matrix_range<viennacl::matrix<T> > data();
        viennacl::matrix<T> matrix() {
            return A;
        }
        
};

#endif
