
#include <iostream>

#include "gpuR/windows_check.hpp"
#include "gpuR/dynEigenMat.hpp"


template<typename T>
dynEigenMat<T>::dynEigenMat(SEXP A_){
    A = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A_);
    orig_nr = A.rows();
    orig_nc = A.cols();
    nr = A.rows();
    nc = A.cols();
    r_start = 1;
    r_end = nr;
    c_start = 1;
    c_end = nc;
    // ptr = A.data();
    // raw_ptr = &A;
    
    // std::cout << "A" << std::endl;
    // std::cout << A << std::endl;
    
    ptr = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A);
    
    // std::cout << "ptr" << std::endl;
    // std::cout << *ptr << std::endl;
    // ptr.reset(raw_ptr);
}

template<typename T>
dynEigenMat<T>::dynEigenMat(Eigen::Matrix<T, Eigen::Dynamic,Eigen::Dynamic> A_){
    A = A_;
    orig_nr = A.rows();
    orig_nc = A.cols();
    nr = A.rows();
    nc = A.cols();
    r_start = 1;
    r_end = nr;
    c_start = 1;
    c_end = nc;
    // ptr = A.data();
    
    // std::cout << "before" << std::endl;
    // std::cout << raw_ptr << std::endl;
    // 
    // raw_ptr = &A;
    //
    // std::cout << "after" << std::endl;
    // std::cout << raw_ptr << std::endl;
    
    ptr = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A);
    
    // std::cout << "check smart pointer" << std::endl;
    // std::cout << *ptr << std::endl;
    
}

template<typename T>
dynEigenMat<T>::dynEigenMat(int nr_in, int nc_in){
    A = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(nr_in, nc_in);
    orig_nr = nr_in;
    orig_nc = nc_in;
    nr = nr_in;
    nc = nc_in;
    r_start = 1;
    r_end = nr_in;
    c_start = 1;
    c_end = nc_in;
    // ptr = A.data();
    // raw_ptr = &A;
    ptr = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A);
    // ptr.reset(raw_ptr);
}

template<typename T>
dynEigenMat<T>:: dynEigenMat(T scalar, int nr_in, int nc_in){
    A = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Constant(nr_in, nc_in, scalar);
    orig_nr = nr_in;
    orig_nc = nc_in;
    nr = nr_in;
    nc = nc_in;
    r_start = 1;
    r_end = nr_in;
    c_start = 1;
    c_end = nc_in;
    // ptr = A.data();
    // raw_ptr = &A;
    ptr = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A);
    // ptr.reset(raw_ptr);
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>* 
dynEigenMat<T>::getPtr(){ 
    // return A.data();
    return ptr.get();
}

template<typename T>
std::shared_ptr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> 
dynEigenMat<T>::getHostPtr(){ 
    return ptr;
}

template<typename T>
viennacl::matrix<T>* 
dynEigenMat<T>::getDevicePtr(){ 
    return shptr.get(); 
}

template<typename T>
viennacl::matrix<T>
dynEigenMat<T>::getDeviceData(){
    return *vclA;
}

// template<typename T>
// void 
// dynEigenMat<T>::setRange(
//         int row_start, int row_end,
//         int col_start, int col_end
// ){
//     r_start = row_start;
//     r_end = row_end;
//     c_start = col_start;
//     c_end = col_end;
// }


template<typename T>
void
dynEigenMat<T>::updateDim(){
    nr = r_end - r_start + 1;
    nc = c_end - c_start + 1;
}

template<typename T>
void
dynEigenMat<T>::setSourceDim(const int rows, const int cols){
    orig_nr = rows;
    orig_nc = cols;
}


template<typename T>
void
dynEigenMat<T>::setMatrix(Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > &Mat){
    A = Mat;
    ptr = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A);
}

template<typename T>
void
dynEigenMat<T>::setMatrix(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &Mat){
    A = Mat;
    ptr = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A);
}

// note - this will assume complete control of pointer and delete it as well
// following deletion of wherever this is set
template<typename T>
void
dynEigenMat<T>::setPtr(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>* ptr_){
    // ptr = ptr_;
    
    // raw_ptr = ptr_;
    // ptr = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A);
    // *raw_ptr = *ptr_;
    ptr.reset(ptr_);
}

template<typename T>
void
dynEigenMat<T>::setHostPtr(std::shared_ptr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> ptr_){
    ptr = ptr_;
}

template<typename T>
Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >
dynEigenMat<T>::data(){
    // Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>* eigen_ptr = ptr.get();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > temp(ptr.get()->data(), orig_nr, orig_nc);
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > block = temp.block(r_start-1, c_start-1, r_end-r_start + 1, c_end-c_start + 1);
    return block;
}

template<typename T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > 
dynEigenMat<T>::matrix(){
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > mat(ptr.get()->data(), nr, nc);
    return mat;
}


template<typename T>
viennacl::matrix<T> 
dynEigenMat<T>::device_data(long ctx_id){
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
}

template<typename T>
void
dynEigenMat<T>::to_device(long ctx_id){
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
    //shptr = std::make_shared<viennacl::matrix<T> >(vclA);
    
    // viennacl::matrix<T> vclMat(K,M, ctx = ctx);
    viennacl::copy(block, *shptr.get());
}

template<typename T>
void
dynEigenMat<T>::to_host(viennacl::matrix<T> &vclMat){
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > temp(ptr.get()->data(), orig_nr, orig_nc);
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ref = temp.block(r_start-1, c_start-1, r_end-r_start + 1, c_end-c_start + 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::OuterStride<> > block(
            ref.data(), ref.rows(), ref.cols(),
            Eigen::OuterStride<>(ref.outerStride())
    );
    
    viennacl::copy(vclMat, block);  
}


template<typename T>
void
dynEigenMat<T>::to_host(){
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > temp(ptr.get()->data(), orig_nr, orig_nc);
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ref = temp.block(r_start-1, c_start-1, r_end-r_start + 1, c_end-c_start + 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::OuterStride<> > block(
            ref.data(), ref.rows(), ref.cols(),
            Eigen::OuterStride<>(ref.outerStride())
    );
    
    viennacl::copy(*shptr.get(), block);  
}

template<typename T>
void
dynEigenMat<T>::release_device(){
    shptr.reset();
}


template class dynEigenMat<int>;
template class dynEigenMat<float>;
template class dynEigenMat<double>;

