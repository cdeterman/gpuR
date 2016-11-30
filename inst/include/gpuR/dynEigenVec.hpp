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
        std::shared_ptr<viennacl::vector_base<T> > shptr;
        
    public:
        Eigen::Matrix<T, Eigen::Dynamic, 1> A;
        viennacl::vector_base<T> *vclA = new viennacl::vector_base<T>();
        
        dynEigenVec() { } // private default constructor
        dynEigenVec(SEXP A_){
            A = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A_);
            size = A.size();
            begin = 1;
            last = size;
            ptr = A.data();
        }
        dynEigenVec(Eigen::Matrix<T, Eigen::Dynamic,1> &A_){
            A = A_;
            size = A.size();
            begin = 1;
            last = size;
            ptr = A.data();
        };
        // dynEigenVec(Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > &A_, int size_){
        //     A = A_;
        //     size = A.size();
        //     begin = 1;
        //     last = size;
        //     ptr = A.data();
        // }
        dynEigenVec(int size_in){
            A = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(size_in);
            size = size_in;
            begin = 1;
            last = size;
            ptr = A.data();
        }
        // dynEigenVec(Eigen::Matrix<T, Eigen::Dynamic,1> &A_, const int start, const int end){
        //     A = A_;
        //     size = A.size();
        //     begin = start - 1;
        //     last = end - 1;
        //     ptr = A.data();
        // }
        // dynEigenVec(Rcpp::XPtr<dynEigenVec<T> > dynVec){
        //     size = dynVec->length();
        //     begin = dynVec->start();
        //     last = dynVec->end();
        //     ptr = dynVec->getPtr();
        // }
        
        T* getPtr() { return A.data(); }
        viennacl::vector_base<T>* getDevicePtr() { return shptr.get(); }
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
        // void to_host(viennacl::vector<T> &vclMat){
        //     // Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > temp(ptr, size, 1);
        //     // Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > block(&temp(begin-1), last - begin + 1);
        //     viennacl::copy(vclMat, A);
        // }
        // copy to device (w/in class)
        
        void to_device(long ctx_id){
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > temp(ptr, size, 1);
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > block(&temp(begin-1), last - begin + 1);
            
            const int M = block.size();
            
            viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
            
            *vclA = viennacl::vector_base<T>(M, ctx=ctx);
            shptr.reset(vclA);
            
            viennacl::fast_copy(block.data(), block.data() + block.size(), shptr.get()->begin());
            // viennacl::copy(block, *shptr.get());
        };
        
        // release device memory
        void release_device(){
            shptr.reset();
        };
        
};

#endif
