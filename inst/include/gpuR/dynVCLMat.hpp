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
#include "viennacl/ocl/backend.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"

#include <RcppEigen.h>

#include <memory>

template <class T> 
class dynVCLMat {
    private:
        int nr, nc;
        viennacl::range row_r;
        viennacl::range col_r;
        viennacl::matrix<T> *ptr;
        std::shared_ptr<viennacl::matrix<T> > shptr;
        // = std::make_shared<viennacl::matrix<T> >();
    
    public:
        viennacl::matrix<T> A;
        
        dynVCLMat() { } // private default constructor
	    dynVCLMat(viennacl::matrix<T> mat, int ctx_id);
        dynVCLMat(SEXP A_, int ctx_id);
        dynVCLMat(
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Am,
            int nr_in, int nc_in, int ctx_id
            );
        dynVCLMat(int nr_in, int nc_in, int ctx_id);
        dynVCLMat(int nr_in, int nc_in, T scalar, int ctx_id);
        dynVCLMat(Rcpp::XPtr<dynVCLMat<T> > dynMat);
        
        viennacl::matrix<T>* getPtr();
        std::shared_ptr<viennacl::matrix<T> > sharedPtr();
        
        int nrow() { return nr; }
        int ncol() { return nc; }
        viennacl::range row_range() { return row_r; }
        viennacl::range col_range() { return col_r; }
        
        void setRange(
            int row_start, int row_end,
            int col_start, int col_end
            );
        void setMatrix(viennacl::matrix<T> mat);
        void setMatrix(viennacl::matrix_range<viennacl::matrix<T> > mat);
	    void createMatrix(int nr_in, int nc_in, int ctx_id);
        void setDims(int nr_in, int nc_in){
            nr = nr_in;
            nc = nc_in;
        }
        
        void setPtr(viennacl::matrix<T>* ptr_);
        void setSharedPtr(std::shared_ptr<viennacl::matrix<T> > ptr_);
        
        viennacl::matrix_range<viennacl::matrix<T> > data();
        viennacl::matrix<T> matrix() {
            return A;
        }
        
};

#endif
