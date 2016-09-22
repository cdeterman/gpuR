
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
        dynEigenMat(SEXP A_){
            A = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A_);
            orig_nr = A.rows();
            orig_nc = A.cols();
            nr = A.rows();
            nc = A.cols();
            r_start = 1;
            r_end = nr;
            c_start = 1;
            c_end = nc;
            
            ptr = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A);
        };
        dynEigenMat(Eigen::Matrix<T, Eigen::Dynamic,Eigen::Dynamic> A_){
            A = A_;
            orig_nr = A.rows();
            orig_nc = A.cols();
            nr = A.rows();
            nc = A.cols();
            r_start = 1;
            r_end = nr;
            c_start = 1;
            c_end = nc;
            
            ptr = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A);
        };
        dynEigenMat(int nr_in, int nc_in){
            A = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(nr_in, nc_in);
            orig_nr = nr_in;
            orig_nc = nc_in;
            nr = nr_in;
            nc = nc_in;
            r_start = 1;
            r_end = nr_in;
            c_start = 1;
            c_end = nc_in;
            ptr = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A);
        };
        dynEigenMat(T scalar, int nr_in, int nc_in){
            A = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Constant(nr_in, nc_in, scalar);
            orig_nr = nr_in;
            orig_nc = nc_in;
            nr = nr_in;
            nc = nc_in;
            r_start = 1;
            r_end = nr_in;
            c_start = 1;
            c_end = nc_in;
            ptr = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A);
        };
        dynEigenMat(
            Eigen::Matrix<T, Eigen::Dynamic,Eigen::Dynamic> &A_, 
            const int row_start, const int row_end,
            const int col_start, const int col_end
            );
        dynEigenMat(Rcpp::XPtr<dynEigenMat<T> > dynMat);
        
        // pointer access
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>* getPtr(){
            return ptr.get();
        };
        std::shared_ptr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> getHostPtr(){
            return ptr;
        };
        viennacl::matrix<T>* getDevicePtr(){
            return shptr.get(); 
        };
        
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
        void updateDim(){
            nr = r_end - r_start + 1;
            nc = c_end - c_start + 1;
        };
        void setSourceDim(const int rows, const int cols){
            orig_nr = rows;
            orig_nc = cols;
        };
        
        // setting matrix explicitly
        void setMatrix(Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > &Mat){
            A = Mat;
            ptr = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A);
        };
        void setMatrix(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &Mat){
            A = Mat;
            ptr = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A);
        };
        
        // setting pointer explicitly
        // note - this will assume complete control of pointer and delete it as well
        // following deletion of wherever this is set
        void setPtr(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>* ptr_){
            ptr.reset(ptr_);
        };
        void setHostPtr(std::shared_ptr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> ptr_){
            ptr = ptr_;
        };
        
        // get host data
        Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > data(){
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > temp(ptr.get()->data(), orig_nr, orig_nc);
            Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > block = temp.block(r_start-1, c_start-1, r_end-r_start + 1, c_end-c_start + 1);
            return block;
        };
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > matrix(){
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > mat(ptr.get()->data(), nr, nc);
            return mat;
        };
        
        // get device data
        viennacl::matrix<T> device_data(long ctx_id){
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > temp(ptr.get()->data(), orig_nr, orig_nc);
            Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ref = temp.block(r_start-1, c_start-1, r_end-r_start + 1, c_end-c_start + 1);
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::OuterStride<> > block(
                    ref.data(), ref.rows(), ref.cols(),
                    Eigen::OuterStride<>(ref.outerStride())
            );
            
            const int M = block.cols();
            const int K = block.rows();
            
            viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
            
            viennacl::matrix<T> vclMat(K,M, ctx = ctx);
            viennacl::copy(block, vclMat);
            
            return vclMat;
        };
        viennacl::matrix<T> getDeviceData(){
            return *vclA;
        };
        
        // copy to device (w/in class)
        void to_device(long ctx_id){
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > temp(ptr.get()->data(), orig_nr, orig_nc);
            Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ref = temp.block(r_start-1, c_start-1, r_end-r_start + 1, c_end-c_start + 1);
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::OuterStride<> > block(
                    ref.data(), ref.rows(), ref.cols(),
                    Eigen::OuterStride<>(ref.outerStride())
            );
            
            const int M = block.cols();
            const int K = block.rows();
            
            viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
            
            *vclA = viennacl::matrix<T>(K, M, ctx=ctx);
            shptr.reset(vclA);
            
            viennacl::copy(block, *shptr.get());
        };
        
        // release device memory
        void release_device(){
            shptr.reset();
        };
        
        // copy back to host
        void to_host(){
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > temp(ptr.get()->data(), orig_nr, orig_nc);
            Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ref = temp.block(r_start-1, c_start-1, r_end-r_start + 1, c_end-c_start + 1);
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::OuterStride<> > block(
                    ref.data(), ref.rows(), ref.cols(),
                    Eigen::OuterStride<>(ref.outerStride())
            );
            
            viennacl::copy(*shptr.get(), block);  
        };
        void to_host(viennacl::matrix<T> &vclMat){
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > temp(ptr.get()->data(), orig_nr, orig_nc);
            Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ref = temp.block(r_start-1, c_start-1, r_end-r_start + 1, c_end-c_start + 1);
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::OuterStride<> > block(
                    ref.data(), ref.rows(), ref.cols(),
                    Eigen::OuterStride<>(ref.outerStride())
            );
            
            viennacl::copy(vclMat, block);  
        };
};

template class dynEigenMat<int>;
template class dynEigenMat<float>;
template class dynEigenMat<double>;

#endif
