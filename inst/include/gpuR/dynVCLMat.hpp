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
        Rcpp::StringVector _colNames, _rowNames;
        viennacl::range row_r;
        viennacl::range col_r;
        viennacl::matrix<T> *ptr;
        std::shared_ptr<viennacl::matrix<T> > shptr;
        // = std::make_shared<viennacl::matrix<T> >();
    
    public:
        viennacl::matrix<T> A;
        
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
	        A.switch_memory_context(ctx);
	        A = mat;
	        
	        nr = A.size1();
	        nc = A.size2();
	        ptr = &A;
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
            
            A.switch_memory_context(ctx);
            A = viennacl::matrix<T>(K,M, ctx);
            
            viennacl::copy(Am, A); 
            
            nr = K;
            nc = M;
            ptr = &A;
            
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
            
            A.switch_memory_context(ctx);
            A = viennacl::matrix<T>(nr_in, nc_in, ctx);
            viennacl::copy(Am, A); 
            
            nr = nr_in;
            nc = nc_in;
            ptr = &A;
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
            
            A.switch_memory_context(ctx);
            A = viennacl::zero_matrix<T>(nr_in, nc_in, ctx);
            
            nr = nr_in;
            nc = nc_in;
            ptr = &A;
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
            
            A.switch_memory_context(ctx);
            A = viennacl::scalar_matrix<T>(nr_in, nc_in, scalar, ctx);
            
            nr = nr_in;
            nc = nc_in;
            ptr = &A;
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
            A = mat;
            ptr = &A;
            shptr = std::make_shared<viennacl::matrix<T> >(A);
            // shptr.reset(ptr);
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
            ptr = ptr_;
            shptr = std::make_shared<viennacl::matrix<T> >(*ptr);
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
            return A;
        }
        
        viennacl::vector<T> row(int row_id) {
            // always refer to the block
            viennacl::matrix_range<viennacl::matrix<T> > m_sub(*shptr.get(), row_r, col_r);
            
            // return the desired row
            return viennacl::row(m_sub, row_id);
        }
        
};

#endif
