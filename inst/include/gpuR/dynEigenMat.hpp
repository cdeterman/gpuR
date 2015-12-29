#pragma once
#ifndef DYNEIGEN_MAT_HPP
#define DYNEIGEN_MAT_HPP

#include <RcppEigen.h>

template <class T> 
class dynEigenMat {
    private:
        int nr, orig_nr, nc, orig_nc, r_start, r_end, c_start, c_end;
        T* ptr;
        
    public:
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A;
//        Eigen::Block<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > block;
        
        dynEigenMat() { }; // private default constructor
        dynEigenMat(SEXP A_);
        dynEigenMat(Eigen::Matrix<T, Eigen::Dynamic,Eigen::Dynamic> &A_);
        dynEigenMat(int nr_in, int nc_in);
        dynEigenMat(
            Eigen::Matrix<T, Eigen::Dynamic,Eigen::Dynamic> &A_, 
            const int row_start, const int row_end,
            const int col_start, const int col_end
            );
        dynEigenMat(Rcpp::XPtr<dynEigenMat<T> > dynMat);
        
        T* getPtr() { return A.data(); }
        int nrow() { return nr; }
        int ncol() { return nc; }
        int row_start() { return r_start; }
        int row_end() { return r_end; }
        int col_start() { return c_start; }
        int col_end() { return c_end; }
        void setRange(
            int row_start, int row_end,
            int col_start, int col_end
            ){
            r_start = row_start;
            r_end = row_end;
            c_start = col_start;
            c_end = col_end;
        }
        void updateDim(){
            nr = r_end - r_start + 1;
            nc = c_end - c_start + 1;
        }
        void setSourceDim(const int rows, const int cols){
            orig_nr = rows;
            orig_nc = cols;
        }
        void setMatrix(Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > &Mat){
            A = Mat;
        }
        void setMatrix(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &Mat){
            A = Mat;
        }
        void setPtr(T* ptr_){
            ptr = ptr_;
        }
        Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > data();
        //Eigen::Matrix<T, Eigen::Dynamic, 1> data() { return A; }
//        Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > data() { 
//            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > temp(ptr, nr, nc);
//////            Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > block(&temp(c_start*c_end-1), last - begin + 1);
//            Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > block = temp.block(r_start-1, c_start-1, r_end-r_start + 1, c_end-c_start + 1);
//            std::cout << "internal block" << std::endl;
//            std::cout << block << std::endl;
////            std::cout << nr << nc << r_start << c_start << r_end << c_end << r_end-r_start + 1 << c_end - c_start + 1 << std::endl; 
////            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > block(block_temp.data(), block_temp.rows(), block_temp.cols());
//            return block;
////            return temp;
//        }
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > matrix() {
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > mat(ptr, nr, nc);
//            Eigen::Matrix<T, Eigen::Dynamic, 1>& vec = A;
            return mat;
        }
        
};

#endif
