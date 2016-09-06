

#include "gpuR/windows_check.hpp"
#include "gpuR/dynVCLVec.hpp"

template<typename T>
dynVCLVec<T>::dynVCLVec(
    SEXP A_,
    int ctx_id)
{
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

template<typename T>
dynVCLVec<T>::dynVCLVec(
    int size_in,
    int ctx_id)
{
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

template<typename T>
dynVCLVec<T>::dynVCLVec(
    viennacl::vector<T> vec,
    int ctx_id)
{
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


// template<typename T>
// dynVCLVec<T>::dynVCLVec(Rcpp::XPtr<dynVCLVec<T> > dynVec)
// {
//     size = dynVec->length();
//     begin = dynVec->start();
//     last = dynVec->end();
//     ptr = dynVec->getPtr();
//     viennacl::range temp_r(begin-1, last);
//     r = temp_r;
// }

template<typename T>
void 
dynVCLVec<T>::setRange(int start, int end){
    viennacl::range temp_r(start-1, end);
    r = temp_r;
    begin = start;
    last = end;
}

template<typename T>
void 
dynVCLVec<T>::setPtr(viennacl::vector<T>* ptr_){
    ptr = ptr_;
}

template<typename T>
viennacl::vector_range<viennacl::vector<T> >
dynVCLVec<T>::data() { 
    viennacl::vector_range<viennacl::vector<T> > v_sub(*ptr, r);
    return v_sub;
}

template<typename T>
void 
dynVCLVec<T>::updateSize(){
    size = last - begin;
}

template<typename T>
void
dynVCLVec<T>::setVector(viennacl::vector<T> vec){
    A = vec;
    ptr = &A;
}

template class dynVCLVec<int>;
template class dynVCLVec<float>;
template class dynVCLVec<double>;

