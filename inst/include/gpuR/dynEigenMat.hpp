
#ifndef DYNEIGEN_MAT_HPP
#define DYNEIGEN_MAT_HPP

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

// ViennaCL headers
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"

#include <RcppEigen.h>

#include <memory>

template <class T> 
class dynEigenMat {
    private:
        int nr, orig_nr, nc, orig_nc, r_start, r_end, c_start, c_end;
        // T* ptr;
        // Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>* raw_ptr;
        std::shared_ptr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ptr;
        std::shared_ptr<viennacl::matrix<T> > shptr;
        
    public:
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A;
        viennacl::matrix<T> *vclA = new viennacl::matrix<T>();
        // Eigen::Block<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > block;
        
        
        // initializers
        dynEigenMat() { }; // private default constructor
        dynEigenMat(SEXP A_);
        dynEigenMat(Eigen::Matrix<T, Eigen::Dynamic,Eigen::Dynamic> A_);
        dynEigenMat(int nr_in, int nc_in);
        dynEigenMat(T scalar, int nr_in, int nc_in);
        dynEigenMat(
            Eigen::Matrix<T, Eigen::Dynamic,Eigen::Dynamic> &A_, 
            const int row_start, const int row_end,
            const int col_start, const int col_end
            );
        dynEigenMat(Rcpp::XPtr<dynEigenMat<T> > dynMat);
        
        // pointer access
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>* getPtr();
        std::shared_ptr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> getHostPtr();
        viennacl::matrix<T>* getDevicePtr();
        
        // object values
        int nrow(){ return nr; }
        int ncol(){ return nc; }
        int row_start(){ return r_start; }
        int row_end(){ return r_end; }
        int col_start(){ return c_start; }
        int col_end(){ return c_end; }
        void setRange(
            int row_start, int row_end,
            int col_start, int col_end
        ){
            r_start = row_start;
            r_end = row_end;
            c_start = col_start;
            c_end = col_end;
        };
        void updateDim();
        void setSourceDim(const int rows, const int cols);
        
        // setting matrix explicitly
        void setMatrix(Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > &Mat);
        void setMatrix(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &Mat);
        
        // setting pointer explicitly
        void setPtr(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>* ptr_);
        void setHostPtr(std::shared_ptr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> ptr_);
        
        // get host data
        Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > data();
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > matrix();
        
        // get device data
        viennacl::matrix<T> device_data(long ctx_id);
        viennacl::matrix<T> getDeviceData();
        
        // copy to device (w/in class)
        void to_device(long ctx_id);
        
        // release device memory
        void release_device();
        
        // copy back to host
        void to_host();
        void to_host(viennacl::matrix<T> &vclMat);
};

#endif
