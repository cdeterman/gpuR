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
#include "viennacl/matrix.hpp"

#include <RcppEigen.h>

#include <memory>

// template<typename T>
// struct D { 
//     void operator()(auto* p) const {
//         std::cout << "Call delete from function object...\n";
//         delete p;
//     }
// };

template <class T> 
class dynVCLVec {
    private:
        bool shared;
        int shared_type;
        viennacl::range r;
        // viennacl::vector_base<T> *ptr;
        std::shared_ptr<viennacl::vector_base<T> > shptr;
        // std::shared_ptr<viennacl::vector_base<T> > shptr(std::nullptr_t, D());
        
        viennacl::matrix<T> *ptr_matrix;
    
    protected:
        viennacl::vector_range<viennacl::vector_base<T> > sharedVector(){
            viennacl::vector_base<T> tmp(ptr_matrix->handle(), ptr_matrix->internal_size(), 0, 1);
            viennacl::vector_range<viennacl::vector_base<T> > v_sub(tmp, r);
            return v_sub;
        }
        
        viennacl::vector_range<viennacl::vector_base<T> > sharedRow(){
            // viennacl::vector_base<T> tmp(ptr_matrix->handle(), ptr_matrix->internal_size(), 0, 1);
            viennacl::vector_base<T> tmp(ptr_matrix->handle(), ptr_matrix->size2(), begin, 1);
            viennacl::vector_range<viennacl::vector_base<T> > v_sub(tmp, r);
            return v_sub;
        }
        
        viennacl::vector_range<viennacl::vector_base<T> > sharedCol(){
            // viennacl::vector_base<T> tmp(ptr_matrix->handle(), ptr_matrix->internal_size(), 0, 1);
            
            // std::cout << "returning column" << std::endl;
            viennacl::vector_base<T> tmp(ptr_matrix->handle(), ptr_matrix->size1(), begin, ptr_matrix->internal_size2());
            // std::cout << "got column" << std::endl;
            
            viennacl::vector_range<viennacl::vector_base<T> > v_sub(tmp, r);
            // std::cout << "got range" << std::endl;
            
            return v_sub;
        }
    
    public:
        // viennacl::vector_base<T> A;
    	int size,begin,last;
        
        ~ dynVCLVec() {
            this->release_device();
        }
        dynVCLVec() { } // private default constructor
        // dynVCLVec(viennacl::vector_base<T> *vec) : ptr(vec){
        //     // A = vec;
        // 
        //     size = vec->size();
        //     begin = 1;
        //     last = size;
        //     viennacl::range temp_r(0, size);
        //     r = temp_r;
        //     shptr = std::make_shared<viennacl::vector_base<T> >(A);
        // }
        dynVCLVec(viennacl::matrix<T> *mat) : ptr_matrix(mat) {
            shared = true;
            shared_type = 0;
            size = mat->internal_size();
            begin = 1;
            last = size;
            viennacl::range temp_r(0, size);
            r = temp_r;
        }
        // margin - true = rows, false = cols
        dynVCLVec(viennacl::matrix<T> *mat, const bool margin, int start) : ptr_matrix(mat) {
            shared = true;
            if(margin){
                shared_type = 1;
                size = mat->size2();
            }else{
                shared_type = 2;
                size = mat->size1();
                // viennacl::vector_base<T> A = viennacl::vector_base<T>(ptr_matrix->handle(), ptr_matrix->size1(), begin, ptr_matrix->internal_size2());
                // ptr = &A;
            }
            begin = start;
            last = size;
            viennacl::range temp_r(0, size);
            r = temp_r;
        }
        dynVCLVec(viennacl::matrix_range<viennacl::matrix<T> > &mat, const int ctx_id) {
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            viennacl::vector_base<T> A = viennacl::vector_base<T>(mat.size1() * mat.size2(), ctx); 
            
            viennacl::matrix_base<T> dummy(A.handle(),
                                           mat.size1(), 0, 1, mat.size1(),   //row layout
                                           mat.size2(), 0, 1, mat.size2(),   //column layout
                                           true); // row-major
            dummy = mat;
            
            // shared = true;
            size = A.size();
            begin = 1;
            last = size;
            // ptr = &A;
            viennacl::range temp_r(0, size);
            r = temp_r;
            
            shared = false;
            shared_type = 0;
            shptr = std::make_shared<viennacl::vector_base<T> >(A);
        }
        dynVCLVec(viennacl::vector_base<T> vec, int ctx_id) {
            // viennacl::context ctx;
            // 
            // // // explicitly pull context for thread safe forking
            // ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            // 
            // A.switch_memory_context(ctx);
            viennacl::vector_base<T> A = vec;

            size = A.size();
            begin = 1;
            last = size;
            // ptr = &A;
            viennacl::range temp_r(0, size);
            r = temp_r;
            
            shared = false;
            shared_type = 0;
            shptr = std::make_shared<viennacl::vector_base<T> >(A);
        }
        // dynVCLVec(viennacl::vector_range<viennacl::vector_base<T> > vec, int ctx_id){
        //     viennacl::context ctx;
        // 
        //     // explicitly pull context for thread safe forking
        //     ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
        // 
        //     A.switch_memory_context(ctx);
        //     A = vec;
        // 
        //     size = A.size();
        //     begin = 1;
        //     last = size;
        //     ptr = &A;
        //     viennacl::range temp_r(0, size);
        //     r = temp_r;
        // }
        dynVCLVec(SEXP A_, int ctx_id) {
            Eigen::Matrix<T, Eigen::Dynamic, 1> Am;
            Am = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A_);
            
            int K = Am.size();
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            // std::cout << "about to initialize" << std::endl;
            viennacl::vector_base<T> A = viennacl::vector_base<T>(K, ctx); 
            // std::cout << "initialized vector" << std::endl;
            
            viennacl::fast_copy(Am.data(), Am.data() + Am.size(), A.begin());
            // viennacl::fast_copy(Am.begin(), Am.end(), A.begin());
            
            // std::cout << "copied" << std::endl;
            
            size = A.size();
            begin = 1;
            last = size;
            // ptr = &A;
            viennacl::range temp_r(0, size);
            r = temp_r;
            
            shared = false;
            shared_type = 0;
            shptr = std::make_shared<viennacl::vector_base<T> >(A);
        }
        // dynVCLVec(Eigen::Matrix<T, Eigen::Dynamic,1> &A_);
        // dynVCLVec(Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > &A_, int size_);
        dynVCLVec(const int size_in, const int ctx_id) {
            
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            viennacl::vector_base<T> A = viennacl::vector_base<T>(size_in, ctx);
            // A = viennacl::zero_vector<T>(size_in, ctx);
            // A = static_cast<viennacl::vector_base<T> >(A);
            
            viennacl::linalg::vector_assign(A, (T)(0));
            
            begin = 1;
            last = size_in;
            // ptr = &A;
            viennacl::range temp_r(begin-1, last);
            r = temp_r;
            
            shared = false;
            shared_type = 0;
            shptr = std::make_shared<viennacl::vector_base<T> >(A);
        }
        dynVCLVec(const int size_in, T scalar, const int ctx_id){
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            // A.switch_memory_context(ctx);
            // A = viennacl::scalar_vector<T>(size_in, scalar, ctx);
            // A = static_cast<viennacl::vector_base<T> >(A);
            
            viennacl::vector_base<T> A = viennacl::vector_base<T>(size_in, ctx);
            viennacl::linalg::vector_assign(A, scalar);
            
            size = size_in;
            // ptr = &A;
            begin = 1;
            last = size_in;
            viennacl::range temp_r(begin-1, last);
            r = temp_r;
            // shptr.reset(&A, [](decltype(&A) p) {
            // 	std::cout << "Call delete from lambda...\n";
            // 	// delete p;
            // });
            
            shared = false;
            shared_type = 0;
            shptr = std::make_shared<viennacl::vector_base<T> >(A);
        };
        // dynVCLVec(Eigen::Matrix<T, Eigen::Dynamic,1> &A_, const int start, const int end);
        // dynVCLVec(Rcpp::XPtr<dynVCLVec<T> > dynVec){
        //     size = dynVec->length();
        //     begin = dynVec->start();
        //     last = dynVec->end();
        //     ptr = dynVec->getPtr();
        //     viennacl::range temp_r(begin-1, last);
        //     r = temp_r;
        // }
        
        viennacl::vector_base<T>* getPtr() { return shptr.get(); }
        std::shared_ptr<viennacl::vector_base<T> > sharedPtr() { 
            return shptr; 
        }
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
        // void setVector(viennacl::vector_range<viennacl::vector_base<T> > vec);
        void setVector(viennacl::vector_base<T> vec){
            viennacl::vector_base<T> A = vec;
            // ptr = &A;
            shptr = std::make_shared<viennacl::vector_base<T> >(A);
            
            shared = false;
            shared_type = 0;
        };
        void setPtr(viennacl::vector_base<T>* ptr_){
            // ptr = ptr_;
            // this will result in a copy I believe
            // shptr = std::make_shared<viennacl::vector_base<T> >(*ptr_);
            shptr.reset(ptr_);
            
            shared = false;
            shared_type = 0;
        };
        void setSharedPtr(std::shared_ptr<viennacl::vector_base<T> > shptr_){
            shptr = shptr_;
            
            shared = false;
            shared_type = 0;
        };
        
        viennacl::vector_range<viennacl::vector_base<T> > data(){ 
            if(this->isShared()){
                switch(shared_type){
                    case 0:
                        return this->sharedVector();
                    case 1:
                        return this->sharedRow();
                    case 2:
                        return this->sharedCol();
                    default:
                        throw Rcpp::exception("unknown shared_type for vclVector object!");
                }
                
            }else{
                // std::cout << "return vector" << std::endl;
                // viennacl::vector_base<T> tmp = *shptr.get();
                // std::cout << tmp << std::endl;
                viennacl::vector_range<viennacl::vector_base<T> > v_sub(*shptr.get(), r);
                return v_sub;
            }
        };
        viennacl::vector_base<T> vector() {
            return *shptr.get();
        }
        viennacl::vector_range<viennacl::vector_base<T> > range(viennacl::range in_range){
            viennacl::vector_range<viennacl::vector_base<T> > v_sub(*shptr.get(), r);
            viennacl::vector_range<viennacl::vector_base<T> > v_out(v_sub, in_range);
            return v_out;
        }
        bool isShared(){
            return shared;
        }
        
        void fill(T scalar){
            viennacl::vector_range<viennacl::vector_base<T> > v_sub(*shptr.get(), r);
            viennacl::linalg::vector_assign(v_sub, scalar);
        }
        
        void fill(viennacl::range in_range, T scalar){
            viennacl::vector_range<viennacl::vector_base<T> > v_sub(*shptr.get(), r);
            viennacl::vector_range<viennacl::vector_base<T> > v_sub2(v_sub, in_range);
            viennacl::linalg::vector_assign(v_sub2, scalar);
        }
        
        void fill(viennacl::slice s, T scalar){
            viennacl::vector_range<viennacl::vector_base<T> > v_sub(*shptr.get(), r);
            viennacl::vector_slice<viennacl::vector_base<T> > v_sub2(v_sub, s);
            viennacl::linalg::vector_assign(v_sub2, scalar);
        }
        
        void fill(viennacl::slice s, viennacl::vector<T> v){
            viennacl::vector_range<viennacl::vector_base<T> > v_sub(*shptr.get(), r);
            viennacl::vector_slice<viennacl::vector_base<T> > v_sub2(v_sub, s);
            v_sub2 = v;
        }
        void fill(Rcpp::IntegerVector idx, SEXP A){
            viennacl::vector_range<viennacl::vector_base<T> > v_sub(*shptr.get(), r);
            
            Eigen::Matrix<T, Eigen::Dynamic, 1> Am;
            Am = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A);
            
            for(int i = 0; i < idx.size(); i++) {
                v_sub(idx[i]) = Am(i);
            }
        }
        
        // release device memory
        void release_device(){
            shptr.reset();
        };
        
};

#endif
