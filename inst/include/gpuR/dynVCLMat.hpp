#pragma once
#ifndef DYNVCL_MAT_HPP
#define DYNVCL_MAT_HPP

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

#include <complex>

// ViennaCL headers
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"

#include <RcppEigen.h>

#include <type_traits>
#include <memory>

template <class T> 
class dynVCLMat {
    
    static_assert(std::is_same<T, double>::value || 
                  std::is_same<T, float>::value ||
                  std::is_same<T, int>::value,
                  "some meaningful error message");
    
    private:
        int nr, nc;
        Rcpp::StringVector _colNames, _rowNames;
        viennacl::range row_r;
        viennacl::range col_r;
        // viennacl::matrix<T> *ptr;
        std::shared_ptr<viennacl::matrix<T> > shptr;
        // = std::make_shared<viennacl::matrix<T> >();
    
    public:
        // viennacl::matrix<T> A;
        
        dynVCLMat() { 
            
            viennacl::range temp_rr(0, 0);
            viennacl::range temp_cr(0, 0);
            
            row_r = temp_rr;
            col_r = temp_cr;
            
        } // private default constructor
	    dynVCLMat(viennacl::matrix<T> mat, int ctx_id){
	        
	        viennacl::context ctx;
	        
	        // explicitly pull context for thread safe forking
	        ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
	        
	        // must explicity switch context to make sure the same
	        // it appears when class initialized the A is set to current context (may not be desired)
	        // A.switch_memory_context(ctx);
	        
	        viennacl::matrix<T> A = mat;
	        
	        nr = A.size1();
	        nc = A.size2();
	        // ptr = &A;
	        shptr = std::make_shared<viennacl::matrix<T> >(A);
	        // shptr.reset(ptr);
	        viennacl::range temp_rr(0, nr);
	        viennacl::range temp_cr(0, nc);
	        row_r = temp_rr;
	        col_r = temp_cr;
	    };
        dynVCLMat(SEXP A_, int ctx_id){
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Am;
            Am = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A_);
            
            int K = Am.rows();
            int M = Am.cols();
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            // A.switch_memory_context(ctx);
            viennacl::matrix<T> A = viennacl::matrix<T>(K,M, ctx);
            
            viennacl::copy(Am, A); 
            
            nr = K;
            nc = M;
            // ptr = &A;
            
            shptr = std::make_shared<viennacl::matrix<T> >(A);
            // shptr.reset(ptr);
            viennacl::range temp_rr(0, nr);
            viennacl::range temp_cr(0, nc);
            row_r = temp_rr;
            col_r = temp_cr;
        };
        dynVCLMat(
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Am,
            int nr_in, int nc_in, int ctx_id
        ){    
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            // A.switch_memory_context(ctx);
            viennacl::matrix<T> A = viennacl::matrix<T>(nr_in, nc_in, ctx);
            viennacl::copy(Am, A); 
            
            nr = nr_in;
            nc = nc_in;
            // ptr = &A;
            shptr = std::make_shared<viennacl::matrix<T> >(A);
            // shptr.reset(ptr);
            viennacl::range temp_rr(0, nr);
            viennacl::range temp_cr(0, nc);
            row_r = temp_rr;
            col_r = temp_cr;
        };
        dynVCLMat(int nr_in, int nc_in, int ctx_id){    
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            // A.switch_memory_context(ctx);
            viennacl::matrix<T> A = viennacl::zero_matrix<T>(nr_in, nc_in, ctx);
            
            nr = nr_in;
            nc = nc_in;
            // ptr = &A;
            shptr = std::make_shared<viennacl::matrix<T> >(A);
            // shptr.reset(ptr);
            viennacl::range temp_rr(0, nr);
            viennacl::range temp_cr(0, nc);
            row_r = temp_rr;
            col_r = temp_cr;
        };
        dynVCLMat(int nr_in, int nc_in, T scalar, int ctx_id){
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            // A.switch_memory_context(ctx);
            viennacl::matrix<T> A = viennacl::scalar_matrix<T>(nr_in, nc_in, scalar, ctx);
            
            nr = nr_in;
            nc = nc_in;
            // ptr = &A;
            shptr = std::make_shared<viennacl::matrix<T> >(A);
            // shptr.reset(ptr);
            viennacl::range temp_rr(0, nr);
            viennacl::range temp_cr(0, nc);
            row_r = temp_rr;
            col_r = temp_cr;
        };
        // dynVCLMat(Rcpp::XPtr<dynVCLMat<T> > dynMat)        {
        //     nr = dynMat->nrow();
        //     nc = dynMat->ncol();
        //     row_r = dynMat->row_range();
        //     col_r = dynMat->col_range();
        //     ptr = dynMat->getPtr();
        // };
        
        
        viennacl::matrix<T>* getPtr(){ 
            return shptr.get(); 
        };
        std::shared_ptr<viennacl::matrix<T> > sharedPtr(){
            return shptr;
        };
        
        int nrow() { return nr; }
        int ncol() { return nc; }
        viennacl::range row_range() { return row_r; }
        viennacl::range col_range() { return col_r; }
        
        void setRange(
            viennacl::range row_in, 
            viennacl::range col_in
        ){
            row_r = row_in;
            col_r = col_in;
        }
        void setRange(
            int row_start, int row_end,
            int col_start, int col_end
        ){
            if(row_r.size() == 0 && col_r.size() == 0){
                viennacl::range temp_rr(row_start, row_end);
                viennacl::range temp_cr(col_start, col_end);
                
                row_r = temp_rr;
                col_r = temp_cr;
            }else{
                viennacl::range temp_rr(row_start + row_r.start(), row_end + row_r.start());
                viennacl::range temp_cr(col_start + col_r.start(), col_end + col_r.start());    
                
                row_r = temp_rr;
                col_r = temp_cr;
            }
        };
        void setMatrix(viennacl::matrix<T> mat){
            viennacl::matrix<T> A = mat;
            // ptr = &A;
            shptr = std::make_shared<viennacl::matrix<T> >(A);
            // shptr.reset(ptr);
        }
        void updateMatrix(const viennacl::matrix<T> &mat){
            viennacl::matrix_range<viennacl::matrix<T> > m_sub(*shptr.get(), row_r, col_r);
            m_sub = mat;
        }
        void updateMatrix(const viennacl::matrix_range<viennacl::matrix<T> > &mat){
            viennacl::matrix_range<viennacl::matrix<T> > m_sub(*shptr.get(), row_r, col_r);
            m_sub = mat;
        }
        // void setMatrix(viennacl::matrix_range<viennacl::matrix<T> > mat){
        //     A = mat;
        //     ptr = &A;
        // };
	    // void createMatrix(int nr_in, int nc_in, int ctx_id){
	    //     
	    //     // std::cout << "creating matrix" << std::endl;
	    //     
	    //     viennacl::context ctx;
	    //     
	    //     // explicitly pull context for thread safe forking
	    //     ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
	    //     
	    //     // std::cout << "pulled context" << std::endl;
	    //     // std::cout << ctx_id << std::endl;
	    //     
	    //     A = viennacl::matrix<T>(nr_in, nc_in, ctx=ctx);
	    //     
	    //     // std::cout << "assigned new matrix" << std::endl;
	    //     
	    //     ptr = &A;
	    // };
        void setDims(int nr_in, int nc_in){
            nr = nr_in;
            nc = nc_in;
        }
        
        void setColumnNames(Rcpp::StringVector names){
            _colNames = names;
        }
        Rcpp::StringVector getColumnNames(){
            return _colNames;
        }
        
        void setPtr(viennacl::matrix<T>* ptr_){
            // ptr = ptr_;
            shptr = std::make_shared<viennacl::matrix<T> >(*ptr_);
            // shptr.reset(ptr_);
        };
        void setSharedPtr(std::shared_ptr<viennacl::matrix<T> > shptr_){
            shptr = shptr_;
        };
        
        viennacl::matrix_range<viennacl::matrix<T> > data(){ 
            viennacl::matrix_range<viennacl::matrix<T> > m_sub(*shptr.get(), row_r, col_r);
            return m_sub;
        };
        viennacl::matrix<T> matrix() {
            return *shptr;
        }
        
        viennacl::vector<T> row(int row_id) {
            // always refer to the block
            viennacl::matrix_range<viennacl::matrix<T> > m_sub(*shptr.get(), row_r, col_r);
            
            // return the desired row
            return viennacl::row(m_sub, row_id);
        }
        
};


template <> 
class dynVCLMat<std::complex<float> > {
        private:
        int nr, nc;
        Rcpp::StringVector _colNames, _rowNames;
        viennacl::range row_r;
        viennacl::range col_r;
        // viennacl::matrix<float> *ptr;
        std::shared_ptr<viennacl::matrix<float> > shptr;
        // = std::make_shared<viennacl::matrix<float> >();
        
        public:
        // viennacl::matrix<float> A;
        
        dynVCLMat() { 
            
            viennacl::range temp_rr(0, 0);
            viennacl::range temp_cr(0, 0);
            
            row_r = temp_rr;
            col_r = temp_cr;
            
        } // private default constructor
        dynVCLMat(viennacl::matrix<float> mat, int ctx_id){
            
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            viennacl::matrix<float> A = mat;
            
            nr = A.size1();
            nc = A.size2();
            shptr = std::make_shared<viennacl::matrix<float> >(A);
            viennacl::range temp_rr(0, nr);
            viennacl::range temp_cr(0, nc);
            row_r = temp_rr;
            col_r = temp_cr;
        };
        dynVCLMat(SEXP A_, int ctx_id){
            Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic> Am;
            Am = Rcpp::as<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic> >(A_);
            
            const int K = Am.rows();
            const int M = Am.cols() * 2;
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            // A.switch_memory_context(ctx);
            viennacl::matrix<float> A = viennacl::zero_matrix<float>(K,M, ctx);
            
            // viennacl::copy(Am, A); 
            
            // assign real elements
            viennacl::matrix_slice<viennacl::matrix<float> > A_sub(A, viennacl::slice(0, 1, K), viennacl::slice(0, 2, M/2));
            viennacl::copy(Am.real(), A_sub);
            
            // assign imaginary elements
            viennacl::matrix_slice<viennacl::matrix<float> > B_sub(A, viennacl::slice(0, 1, K), viennacl::slice(1, 2, M/2));
            viennacl::copy(Am.imag(), B_sub);
            
            nr = K;
            nc = M;
            // ptr = &A;
            
            shptr = std::make_shared<viennacl::matrix<float> >(A);
            // shptr.reset(ptr);
            viennacl::range temp_rr(0, nr);
            viennacl::range temp_cr(0, nc);
            row_r = temp_rr;
            col_r = temp_cr;
        };
        dynVCLMat(
            Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic> Am,
            int nr_in, int nc_in, int ctx_id
        ){    
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            nr = nr_in;
            nc = nc_in * 2;
            
            // A.switch_memory_context(ctx);
            viennacl::matrix<float> A = viennacl::zero_matrix<float>(nr, nc, ctx);
            // viennacl::copy(Am, A); 
            
            // assign real elements
            viennacl::slice s(0, 2, nc);
            viennacl::slice r(0, 1, nr);
            
            // assign existing matrix with new data
            viennacl::matrix_slice<viennacl::matrix<float> > A_sub(A, r, s);
            viennacl::copy(Am.real(), A_sub);
            
            // assign imaginary elements
            s = viennacl::slice(1, 2, nc);
            A_sub = viennacl::matrix_slice<viennacl::matrix<float> >(A, r, s);
            viennacl::copy(Am.imag(), A_sub);
            
            // ptr = &A;
            shptr = std::make_shared<viennacl::matrix<float> >(A);
            // shptr.reset(ptr);
            viennacl::range temp_rr(0, nr);
            viennacl::range temp_cr(0, nc);
            row_r = temp_rr;
            col_r = temp_cr;
        };
        dynVCLMat(int nr_in, int nc_in, int ctx_id){    
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            nr = nr_in;
            nc = nc_in * 2;
            
            // A.switch_memory_context(ctx);
            viennacl::matrix<float> A = viennacl::zero_matrix<float>(nr, nc, ctx);
            
            // ptr = &A;
            shptr = std::make_shared<viennacl::matrix<float> >(A);
            // shptr.reset(ptr);
            viennacl::range temp_rr(0, nr);
            viennacl::range temp_cr(0, nc);
            row_r = temp_rr;
            col_r = temp_cr;
        };
        dynVCLMat(int nr_in, int nc_in, std::complex<float> scalar, int ctx_id){
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            nr = nr_in;
            nc = nc_in * 2;
            
            // A.switch_memory_context(ctx);
            viennacl::matrix<float> A = viennacl::zero_matrix<float>(nr, nc, ctx);
            
            // assign real elements
            viennacl::slice s(0, 2, nc_in);
            viennacl::slice r(0, 1, nr);
            
            // assign existing matrix with new data
            viennacl::matrix_slice<viennacl::matrix<float> > A_sub(A, r, s);
            viennacl::linalg::matrix_assign(A_sub, scalar.real());
            
            // assign imaginary elements
            s = viennacl::slice(1, 2, nc_in);
            A_sub = viennacl::matrix_slice<viennacl::matrix<float> >(A, r, s);
            viennacl::linalg::matrix_assign(A_sub, scalar.imag());
            
            // ptr = &A;
            shptr = std::make_shared<viennacl::matrix<float> >(A);
            // shptr.reset(ptr);
            viennacl::range temp_rr(0, nr);
            viennacl::range temp_cr(0, nc);
            row_r = temp_rr;
            col_r = temp_cr;
        };
        // dynVCLMat(Rcpp::XPtr<dynVCLMat<float> > dynMat)        {
        //     nr = dynMat->nrow();
        //     nc = dynMat->ncol();
        //     row_r = dynMat->row_range();
        //     col_r = dynMat->col_range();
        //     ptr = dynMat->getPtr();
        // };
        
        
        viennacl::matrix<float>* getPtr(){ 
            return shptr.get(); 
        };
        std::shared_ptr<viennacl::matrix<float> > sharedPtr(){
            return shptr;
        };
        
        int nrow() { return nr; }
        int ncol() { return nc; }
        viennacl::range row_range() { return row_r; }
        viennacl::range col_range() { return col_r; }
        
        void setRange(
                viennacl::range row_in, 
                viennacl::range col_in
        ){
            row_r = row_in;
            col_r = col_in;
        }
        void setRange(
                int row_start, int row_end,
                int col_start, int col_end
        ){
            if(row_r.size() == 0 && col_r.size() == 0){
                viennacl::range temp_rr(row_start, row_end);
                viennacl::range temp_cr(col_start, col_end);
                
                row_r = temp_rr;
                col_r = temp_cr;
            }else{
                viennacl::range temp_rr(row_start + row_r.start(), row_end + row_r.start());
                viennacl::range temp_cr(col_start + col_r.start(), col_end + col_r.start());    
                
                row_r = temp_rr;
                col_r = temp_cr;
            }
        };
        void setMatrix(viennacl::matrix<float> mat){
            viennacl::matrix<float> A = mat;
            // ptr = &A;
            shptr = std::make_shared<viennacl::matrix<float> >(A);
            // shptr.reset(ptr);
        }
        void updateMatrix(const viennacl::matrix<float> &mat){
            viennacl::matrix_range<viennacl::matrix<float> > m_sub(*shptr.get(), row_r, col_r);
            m_sub = mat;
        }
        void updateMatrix(const viennacl::matrix_range<viennacl::matrix<float> > &mat){
            viennacl::matrix_range<viennacl::matrix<float> > m_sub(*shptr.get(), row_r, col_r);
            m_sub = mat;
        }
        // void setMatrix(viennacl::matrix_range<viennacl::matrix<float> > mat){
        //     A = mat;
        //     ptr = &A;
        // };
        // void createMatrix(int nr_in, int nc_in, int ctx_id){
        //     
        //     // std::cout << "creating matrix" << std::endl;
        //     
        //     viennacl::context ctx;
        //     
        //     // explicitly pull context for thread safe forking
        //     ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
        //     
        //     // std::cout << "pulled context" << std::endl;
        //     // std::cout << ctx_id << std::endl;
        //     
        //     A = viennacl::matrix<float>(nr_in, nc_in, ctx=ctx);
        //     
        //     // std::cout << "assigned new matrix" << std::endl;
        //     
        //     ptr = &A;
        // };
        void setDims(int nr_in, int nc_in){
            nr = nr_in;
            nc = nc_in * 2;
        }
        
        void setColumnNames(Rcpp::StringVector names){
            _colNames = names;
        }
        Rcpp::StringVector getColumnNames(){
            return _colNames;
        }
        
        void setPtr(viennacl::matrix<float>* ptr_){
            // ptr = ptr_;
            shptr = std::make_shared<viennacl::matrix<float> >(*ptr_);
            // shptr.reset(ptr_);
        };
        void setSharedPtr(std::shared_ptr<viennacl::matrix<float> > shptr_){
            shptr = shptr_;
        };
        
        viennacl::matrix_range<viennacl::matrix<float> > data(){ 
            viennacl::matrix_range<viennacl::matrix<float> > m_sub(*shptr.get(), row_r, col_r);
            return m_sub;
        };
        viennacl::matrix<float> matrix() {
            return *shptr;
        }
        
        viennacl::vector<float> row(int row_id) {
            // always refer to the block
            viennacl::matrix_range<viennacl::matrix<float> > m_sub(*shptr.get(), row_r, col_r);
            
            // return the desired row
            return viennacl::row(m_sub, row_id);
        }
    
};


template <> 
class dynVCLMat<std::complex<double> > {
    private:
        int nr, nc;
        Rcpp::StringVector _colNames, _rowNames;
        viennacl::range row_r;
        viennacl::range col_r;
        // viennacl::matrix<double> *ptr;
        std::shared_ptr<viennacl::matrix<double> > shptr;
        // = std::make_shared<viennacl::matrix<double> >();
        
    public:
        // viennacl::matrix<double> A;
        
        dynVCLMat() { 
            
            viennacl::range temp_rr(0, 0);
            viennacl::range temp_cr(0, 0);
            
            row_r = temp_rr;
            col_r = temp_cr;
            
        } // private default constructor
        dynVCLMat(viennacl::matrix<double> mat, int ctx_id){
            
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            viennacl::matrix<double> A = mat;
            
            nr = A.size1();
            nc = A.size2();
            shptr = std::make_shared<viennacl::matrix<double> >(A);
            viennacl::range temp_rr(0, nr);
            viennacl::range temp_cr(0, nc);
            row_r = temp_rr;
            col_r = temp_cr;
        };
        dynVCLMat(SEXP A_, int ctx_id){
            Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> Am;
            Am = Rcpp::as<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> >(A_);
            
            const int K = Am.rows();
            const int M = Am.cols() * 2;
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            // A.switch_memory_context(ctx);
            viennacl::matrix<double> A = viennacl::zero_matrix<double>(K,M, ctx);

            // assign real elements
            // assign existing matrix with new data
            viennacl::matrix_slice<viennacl::matrix<double> > A_sub(A, viennacl::slice(0, 1, K), viennacl::slice(0, 2, M/2));
            viennacl::copy(Am.real(), A_sub);
            
            // assign imaginary elements
            viennacl::matrix_slice<viennacl::matrix<double> > B_sub(A, viennacl::slice(0, 1, K), viennacl::slice(1, 2, M/2));
            viennacl::copy(Am.imag(), B_sub);

            nr = K;
            nc = M;
            
            shptr = std::make_shared<viennacl::matrix<double> >(A);
            // shptr.reset(ptr);
            viennacl::range temp_rr(0, nr);
            viennacl::range temp_cr(0, nc);
            row_r = temp_rr;
            col_r = temp_cr;
        };
        dynVCLMat(
            Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> Am,
            int nr_in, int nc_in, int ctx_id
        ){    
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            nr = nr_in;
            nc = nc_in * 2;
            
            // A.switch_memory_context(ctx);
            viennacl::matrix<double> A = viennacl::zero_matrix<double>(nr, nc, ctx);
            // viennacl::copy(Am, A); 
            
            // assign real elements
            viennacl::slice s(0, 2, nc);
            viennacl::slice r(0, 1, nr);
            
            // assign existing matrix with new data
            viennacl::matrix_slice<viennacl::matrix<double> > A_sub(A, r, s);
            viennacl::copy(Am.real(), A_sub);
            
            // assign imaginary elements
            s = viennacl::slice(1, 2, nc);
            A_sub = viennacl::matrix_slice<viennacl::matrix<double> >(A, r, s);
            viennacl::copy(Am.imag(), A_sub);
            
            // ptr = &A;
            shptr = std::make_shared<viennacl::matrix<double> >(A);
            // shptr.reset(ptr);
            viennacl::range temp_rr(0, nr);
            viennacl::range temp_cr(0, nc);
            row_r = temp_rr;
            col_r = temp_cr;
        };
        dynVCLMat(int nr_in, int nc_in, int ctx_id){    
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            nr = nr_in;
            nc = nc_in * 2;
            
            // A.switch_memory_context(ctx);
            viennacl::matrix<double> A = viennacl::zero_matrix<double>(nr, nc, ctx);
            
            // ptr = &A;
            shptr = std::make_shared<viennacl::matrix<double> >(A);
            // shptr.reset(ptr);
            viennacl::range temp_rr(0, nr);
            viennacl::range temp_cr(0, nc);
            row_r = temp_rr;
            col_r = temp_cr;
        };
        dynVCLMat(int nr_in, int nc_in, std::complex<double> scalar, int ctx_id){
            viennacl::context ctx;
            
            // explicitly pull context for thread safe forking
            ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
            
            nr = nr_in;
            nc = nc_in;
            
            // A.switch_memory_context(ctx);
            viennacl::matrix<double> A = viennacl::zero_matrix<double>(nr, nc, ctx);
            
            // assign real elements
            viennacl::slice s(0, 2, nc_in);
            viennacl::slice r(0, 1, nr);
            
            // assign existing matrix with new data
            viennacl::matrix_slice<viennacl::matrix<double> > A_sub(A, r, s);
            viennacl::linalg::matrix_assign(A_sub, scalar.real());
            
            // assign imaginary elements
            s = viennacl::slice(1, 2, nc_in);
            A_sub = viennacl::matrix_slice<viennacl::matrix<double> >(A, r, s);
            viennacl::linalg::matrix_assign(A_sub, scalar.imag());
            
            // ptr = &A;
            shptr = std::make_shared<viennacl::matrix<double> >(A);
            // shptr.reset(ptr);
            viennacl::range temp_rr(0, nr);
            viennacl::range temp_cr(0, nc);
            row_r = temp_rr;
            col_r = temp_cr;
        };
        // dynVCLMat(Rcpp::XPtr<dynVCLMat<double> > dynMat)        {
        //     nr = dynMat->nrow();
        //     nc = dynMat->ncol();
        //     row_r = dynMat->row_range();
        //     col_r = dynMat->col_range();
        //     ptr = dynMat->getPtr();
        // };
        
        
        viennacl::matrix<double>* getPtr(){ 
            return shptr.get(); 
        };
        std::shared_ptr<viennacl::matrix<double> > sharedPtr(){
            return shptr;
        };
        
        int nrow() { return nr; }
        int ncol() { return nc; }
        viennacl::range row_range() { return row_r; }
        viennacl::range col_range() { return col_r; }
        
        void setRange(
                viennacl::range row_in, 
                viennacl::range col_in
        ){
            row_r = row_in;
            col_r = col_in;
        }
        void setRange(
                int row_start, int row_end,
                int col_start, int col_end
        ){
            if(row_r.size() == 0 && col_r.size() == 0){
                viennacl::range temp_rr(row_start, row_end);
                viennacl::range temp_cr(col_start, col_end);
                
                row_r = temp_rr;
                col_r = temp_cr;
            }else{
                viennacl::range temp_rr(row_start + row_r.start(), row_end + row_r.start());
                viennacl::range temp_cr(col_start + col_r.start(), col_end + col_r.start());    
                
                row_r = temp_rr;
                col_r = temp_cr;
            }
        };
        void setMatrix(viennacl::matrix<double> mat){
            viennacl::matrix<double> A = mat;
            // ptr = &A;
            shptr = std::make_shared<viennacl::matrix<double> >(A);
            // shptr.reset(ptr);
        }
        void updateMatrix(const viennacl::matrix<double> &mat){
            viennacl::matrix_range<viennacl::matrix<double> > m_sub(*shptr.get(), row_r, col_r);
            m_sub = mat;
        }
        void updateMatrix(const viennacl::matrix_range<viennacl::matrix<double> > &mat){
            viennacl::matrix_range<viennacl::matrix<double> > m_sub(*shptr.get(), row_r, col_r);
            m_sub = mat;
        }
        // void setMatrix(viennacl::matrix_range<viennacl::matrix<double> > mat){
        //     A = mat;
        //     ptr = &A;
        // };
        // void createMatrix(int nr_in, int nc_in, int ctx_id){
        //     
        //     // std::cout << "creating matrix" << std::endl;
        //     
        //     viennacl::context ctx;
        //     
        //     // explicitly pull context for thread safe forking
        //     ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
        //     
        //     // std::cout << "pulled context" << std::endl;
        //     // std::cout << ctx_id << std::endl;
        //     
        //     A = viennacl::matrix<double>(nr_in, nc_in, ctx=ctx);
        //     
        //     // std::cout << "assigned new matrix" << std::endl;
        //     
        //     ptr = &A;
        // };
        void setDims(int nr_in, int nc_in){
            nr = nr_in;
            nc = nc_in * 2;
        }
        
        void setColumnNames(Rcpp::StringVector names){
            _colNames = names;
        }
        Rcpp::StringVector getColumnNames(){
            return _colNames;
        }
        
        void setPtr(viennacl::matrix<double>* ptr_){
            // ptr = ptr_;
            shptr = std::make_shared<viennacl::matrix<double> >(*ptr_);
            // shptr.reset(ptr_);
        };
        void setSharedPtr(std::shared_ptr<viennacl::matrix<double> > shptr_){
            shptr = shptr_;
        };
        
        viennacl::matrix_range<viennacl::matrix<double> > data(){ 
            viennacl::matrix_range<viennacl::matrix<double> > m_sub(*shptr.get(), row_r, col_r);
            return m_sub;
        };
        viennacl::matrix<double> matrix() {
            return *shptr;
        }
        
        viennacl::vector<double> row(int row_id) {
            // always refer to the block
            viennacl::matrix_range<viennacl::matrix<double> > m_sub(*shptr.get(), row_r, col_r);
            
            // return the desired row
            return viennacl::row(m_sub, row_id);
        }
        
};

#endif
