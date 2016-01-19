

#include "gpuR/windows_check.hpp"
#include "gpuR/dynVCLVec.hpp"

template<typename T>
dynVCLVec<T>::dynVCLVec(SEXP A_, int device_flag)
{
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am;
    Am = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A_);
    
    // define device type to use
    if(device_flag == 0){
        //use only GPUs
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
        viennacl::ocl::switch_context(id);
    }else{
        // use only CPUs
        long id = 1;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::cpu_tag());
        viennacl::ocl::switch_context(id);
    }
    
    int K = Am.size();
    
    A = viennacl::vector<T>(K);    
    viennacl::copy(Am, A); 
    
    size = K;
    begin = 1;
    last = size;
    ptr = &A;
    viennacl::range temp_r(0, K);
    r = temp_r;
}

template<typename T>
dynVCLVec<T>::dynVCLVec(int size_in, int device_flag)
{
    // define device type to use
    if(device_flag == 0){
        //use only GPUs
        long id = 0;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
        viennacl::ocl::switch_context(id);
    }else{
        // use only CPUs
        long id = 1;
        viennacl::ocl::set_context_device_type(id, viennacl::ocl::cpu_tag());
        viennacl::ocl::switch_context(id);
    }
    
    A = viennacl::zero_vector<T>(size_in);
    begin = 1;
    last = size_in;
    ptr = &A;
    viennacl::range temp_r(begin-1, last);
    r = temp_r;
}


template<typename T>
dynVCLVec<T>::dynVCLVec(Rcpp::XPtr<dynVCLVec<T> > dynVec)
{
    size = dynVec->length();
    begin = dynVec->start();
    last = dynVec->end();
    ptr = dynVec->getPtr();
    viennacl::range temp_r(begin-1, last);
    r = temp_r;
}

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

template class dynVCLVec<int>;
template class dynVCLVec<float>;
template class dynVCLVec<double>;

