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
        dynVCLVec(viennacl::vector<T> vec, int ctx_id){
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            A.switch_memory_context(ctx);
            A = vec;
            
            size = A.size();
            begin = 1;
            last = size;
            ptr = &A;
            viennacl::range temp_r(0, size);
            r = temp_r;
        }
        dynVCLVec(SEXP A_, int ctx_id){
            Eigen::Matrix<T, Eigen::Dynamic, 1> Am;
            Am = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A_);
            
            int K = Am.size();
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            A = viennacl::vector<T>(K, ctx);    
            viennacl::copy(Am, A); 
            
            size = K;
            begin = 1;
            last = size;
            ptr = &A;
            viennacl::range temp_r(0, K);
            r = temp_r;
        }
        // dynVCLVec(Eigen::Matrix<T, Eigen::Dynamic,1> &A_);
        // dynVCLVec(Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > &A_, int size_);
        dynVCLVec(int size_in, int ctx_id){
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            A = viennacl::zero_vector<T>(size_in, ctx);
            begin = 1;
            last = size_in;
            ptr = &A;
            viennacl::range temp_r(begin-1, last);
            r = temp_r;
        }
        // dynVCLVec(Eigen::Matrix<T, Eigen::Dynamic,1> &A_, const int start, const int end);
        // dynVCLVec(Rcpp::XPtr<dynVCLVec<T> > dynVec){
        //     size = dynVec->length();
        //     begin = dynVec->start();
        //     last = dynVec->end();
        //     ptr = dynVec->getPtr();
        //     viennacl::range temp_r(begin-1, last);
        //     r = temp_r;
        // }
        
        viennacl::vector<T>* getPtr() { return ptr; }
        int length() { return size; }
        int start() { return begin; }
        int end() { return last; }
        void setRange(int start, int end){
            viennacl::range temp_r(start-1, end);
            r = temp_r;
            begin = start;
            last = end;
        }
        void updateSize(){
            size = last - begin;
        };
        // void setVector(viennacl::vector_range<viennacl::vector<T> > vec);
        void setVector(viennacl::vector<T> vec){
            A = vec;
            ptr = &A;
        };
        void setPtr(viennacl::vector<T>* ptr_){
            ptr = ptr_;
        };
        viennacl::vector_range<viennacl::vector<T> > data(){ 
            viennacl::vector_range<viennacl::vector<T> > v_sub(*ptr, r);
            return v_sub;
        };
        viennacl::vector<T> vector() {
            return A;
        }
        viennacl::vector_range<viennacl::vector<T> > range(viennacl::range in_range){
            viennacl::vector_range<viennacl::vector<T> > v_sub(*ptr, r);
            viennacl::vector_range<viennacl::vector<T> > v_out(v_sub, in_range);
            return v_out;
        }
};

#endif
