#pragma once
#ifndef DYNEIGEN_VEC_HPP
#define DYNEIGEN_VEC_HPP

// Use OpenCL with ViennaCL
#ifdef BACKEND_CUDA
#define VIENNACL_WITH_CUDA 1
#elif defined(BACKEND_OPENCL)
#define VIENNACL_WITH_OPENCL 1
#else
#define VIENNACL_WITH_OPENCL 1
#endif

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

#include <RcppEigen.h>

// ViennaCL headers
// #include "viennacl/vector_def.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/vector_proxy.hpp"

#ifndef BACKEND_CUDA
#include "viennacl/ocl/backend.hpp"
#endif

#include <memory>

template <typename T, typename Enable = void>
class dynEigenVec;

template <class T> 
#ifdef BACKEND_CUDA
class dynEigenVec<T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
#else
class dynEigenVec<T>{
#endif
    private:
        int size,begin,last,ctx_id;
        // T* ptr;
        std::shared_ptr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptr;
        std::shared_ptr<viennacl::vector_base<T> > shptr;
        
    public:
        Eigen::Matrix<T, Eigen::Dynamic, 1> A;
        // viennacl::vector_base<T> *vclA = new viennacl::vector_base<T>();
        
        dynEigenVec() { } // private default constructor
        ~dynEigenVec() { } // private default destructor
        dynEigenVec(SEXP A_){
            A = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A_);
            size = A.size();
            begin = 1;
            last = size;
            ptr = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A);
        }
        dynEigenVec(Eigen::Matrix<T, Eigen::Dynamic,1> &A_){
            A = A_;
            size = A.size();
            begin = 1;
            last = size;
            ptr = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A);
        };
        // dynEigenVec(Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > &A_, int size_){
        //     A = A_;
        //     size = A.size();
        //     begin = 1;
        //     last = size;
        //     ptr = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A);
        // }
        dynEigenVec(const int size_in){
            A = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(size_in);
            size = size_in;
            begin = 1;
            last = size;
            ptr = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A);
        }
        dynEigenVec(const int size_in, T scalar){
            A = Eigen::Matrix<T, Eigen::Dynamic, 1>::Constant(size_in, scalar);
            size = size_in;
            begin = 1;
            last = size;
            ptr = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A);
        }
        // dynEigenVec(Eigen::Matrix<T, Eigen::Dynamic,1> &A_, const int start, const int end){
        //     A = A_;
        //     size = A.size();
        //     begin = start - 1;
        //     last = end - 1;
        //     ptr = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A);
        // }
        // dynEigenVec(Rcpp::XPtr<dynEigenVec<T> > dynVec){
        //     size = dynVec->length();
        //     begin = dynVec->start();
        //     last = dynVec->end();
        //     ptr = dynVec->getPtr();
        // }
        
        // pointer access
        // Eigen::Matrix<T, Eigen::Dynamic, 1>* getPtr(){
        //     return ptr.get();
        // };
        std::shared_ptr<Eigen::Matrix<T, Eigen::Dynamic, 1> > getHostPtr(){
            return ptr;
        };
        // viennacl::vector_base<T>* getDevicePtr(){
        //     return shptr.get(); 
        // };
        std::shared_ptr<viennacl::vector_base<T> > getDevicePtr(){
            return shptr; 
        };
        
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
        // void setPtr(T* ptr_){
        //     ptr = ptr_;
        // }
        
        // setting pointer explicitly
        // note - this will assume complete control of pointer and delete it as well
        // following deletion of wherever this is set
        void setPtr(Eigen::Matrix<T, Eigen::Dynamic, 1>* ptr_){
            ptr.reset(ptr_);
        };
        void setHostPtr(std::shared_ptr<Eigen::Matrix<T, Eigen::Dynamic, 1> > ptr_){
            ptr = ptr_;
        };
        
        //Eigen::Matrix<T, Eigen::Dynamic, 1> data() { return A; }
	    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > data() { 
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > temp(ptr.get()->data(), size, 1);
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > block(&temp(begin-1), last - begin + 1);
            return block;
        }
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > vector() {
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > vec(ptr, size, 1);
//            Eigen::Matrix<T, Eigen::Dynamic, 1>& vec = A;
            return vec;
        }

	    // copy back to host
        void to_host(){
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > temp(ptr.get()->data(), size, 1);
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > block(&temp(begin-1), last - begin + 1);
            // viennacl::copy(*shptr.get(), block);
            viennacl::fast_copy(shptr.get()->begin(), shptr.get()->end(), &(block[0]));
        }
	    
	    void to_host(viennacl::vector_base<T> &vclVec){
	        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > temp(ptr.get()->data(), size, 1);
	        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > block(&temp(begin-1), last - begin + 1);
	        viennacl::fast_copy(vclVec.begin(), vclVec.end(), &(block[0]));
	    }
        
	    // copy to device (w/in class)
        void to_device(long ctx_in){
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > temp(ptr.get()->data(), size, 1);
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > block(&temp(begin-1), last - begin + 1);
            
            const int M = block.size();
            ctx_id = ctx_in;
            
#ifdef BACKEND_CUDA
            int cuda_device;
            cudaGetDevice(&cuda_device);
            
            if(ctx_id != cuda_device){
                cudaSetDevice(ctx_id);    
            }
            viennacl::vector_base<T> vclA = viennacl::vector_base<T>(M);
#else            
            viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
            
            viennacl::vector_base<T> vclA = viennacl::vector_base<T>(M, ctx=ctx);
#endif            
            // shptr.reset(vclA);
            shptr = std::make_shared<viennacl::vector_base<T> >(vclA);
            
            viennacl::fast_copy(block.data(), block.data() + block.size(), shptr.get()->begin());
            // viennacl::copy(block, *shptr.get());
        };
        
        // release device memory
        void release_device(){
            shptr.reset();
        };
        
        void setElement(const int idx, SEXP value){
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > MapVec = this->data();
            MapVec(idx-1) = Rcpp::as<T>(value);
        }
        T getElement(const int idx){
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > MapVec = this->data();
            return(MapVec(idx - 1));
        }
};

#endif
