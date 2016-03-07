
#include "gpuR/windows_check.hpp"
#include "gpuR/dynEigenMat.hpp"

template<typename T>
dynEigenMat<T>::dynEigenMat(SEXP A_)
{
    A = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(A_);
    orig_nr = A.rows();
    orig_nc = A.cols();
    nr = A.rows();
    nc = A.cols();
    r_start = 1;
    r_end = nr;
    c_start = 1;
    c_end = nc;
    ptr = A.data();
}

template<typename T>
dynEigenMat<T>::dynEigenMat(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &A_)
{
    A = A_;
    orig_nr = A.rows();
    orig_nc = A.cols();
    nr = A.rows();
    nc = A.cols();
    r_start = 1;
    r_end = nr;
    c_start = 1;
    c_end = nc;
    ptr = A.data();
}

template<typename T>
dynEigenMat<T>::dynEigenMat(Rcpp::XPtr<dynEigenMat<T> > dynMat)
{
    nr = dynMat->nrow();
    nc = dynMat->ncol();
    r_start = dynMat->row_start();
    r_end = dynMat->row_end();
    c_start = dynMat->col_start();
    c_end = dynMat->col_end();
    ptr = dynMat->getPtr();
}

template<typename T>
dynEigenMat<T>::dynEigenMat(int nr_in, int nc_in)
{
    A = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(nr_in, nc_in);
    orig_nr = nr_in;
    orig_nc = nc_in;
    nr = nr_in;
    nc = nc_in;
    r_start = 1;
    r_end = nr_in;
    c_start = 1;
    c_end = nc_in;
    ptr = A.data();
}

template<typename T>
dynEigenMat<T>::dynEigenMat(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &A_,
    const int row_start, const int row_end,
    const int col_start, const int col_end)
{
    A = A_;
    orig_nr = A.rows();
    orig_nc = A.cols();
    nr = A.rows();
    nc = A.cols();
    r_start = row_start-1;
    r_end = row_end-1;
    c_start = col_start-1;
    c_end = col_end-1;
    ptr = A.data();
}

template<typename T>
Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >
dynEigenMat<T>::data() { 
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > temp(ptr, orig_nr, orig_nc);
//    std::cout << "row start: " << r_start << std::endl;
//    std::cout << "col start: " << c_start << std::endl;
//    std::cout << "row end: " << r_end << std::endl;
//    std::cout << "col end: " << c_end << std::endl;
//    std::cout << "row size: " << r_end-r_start + 1 << std::endl;
//    std::cout << "col size: " << c_end-c_start + 1 << std::endl;
    
//    std::cout << "internal full matrix" << std::endl;
//    std::cout << temp << std::endl;
    
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > block = temp.block(r_start-1, c_start-1, r_end-r_start + 1, c_end-c_start + 1);
//    std::cout << "internal block" << std::endl;
//    std::cout << block << std::endl;
    return block;
}

template<typename T>
dynEigenMat<T>::dynEigenMat(T scalar, int nr_in, int nc_in)
{
    A = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Constant(nr_in, nc_in, scalar);
    orig_nr = nr_in;
    orig_nc = nc_in;
    nr = nr_in;
    nc = nc_in;
    r_start = 1;
    r_end = nr_in;
    c_start = 1;
    c_end = nc_in;
    ptr = A.data();
}

template<typename T>
viennacl::matrix<T>
dynEigenMat<T>::device_data() {
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > temp(ptr, orig_nr, orig_nc);
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ref = temp.block(r_start-1, c_start-1, r_end-r_start + 1, c_end-c_start + 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::OuterStride<> > block(
        ref.data(), ref.rows(), ref.cols(),
        Eigen::OuterStride<>(ref.outerStride())
    );
    
    const int M = block.cols();
    const int K = block.rows();
    
    viennacl::matrix<T> vclMat(K,M);
    viennacl::copy(block, vclMat);
    
    return vclMat;
    
}

template<typename T>
void
dynEigenMat<T>::to_host(viennacl::matrix<T> &vclMat) {
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > temp(ptr, orig_nr, orig_nc);
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > ref = temp.block(r_start-1, c_start-1, r_end-r_start + 1, c_end-c_start + 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::OuterStride<> > block(
        ref.data(), ref.rows(), ref.cols(),
        Eigen::OuterStride<>(ref.outerStride())
    );
    
    viennacl::copy(vclMat, block);    
}

template class dynEigenMat<int>;
template class dynEigenMat<float>;
template class dynEigenMat<double>;
