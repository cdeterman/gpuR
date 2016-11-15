#include <RcppEigen.h>

//#include "gpuR/vcl_helpers.hpp"
#include "gpuR/dynVCLMat.hpp"
#include "gpuR/dynVCLVec.hpp"

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::MatrixXi;
using Eigen::VectorXd;
using Eigen::VectorXf;
using Eigen::VectorXi;

using namespace Rcpp;

// template <typename T>
// void
// vclMatTovclVec(SEXP ptrA_, List ptrList){
//     
//     Rcpp::XPtr<dynVCLVec<T> > ptrA(ptrA_);
//     
//     int start = 0;
//     int end;
//     
//     // loop through each matrix in list
//     for(List::iterator it = ptrList.begin(); it != ptrList.end(); ++it){
//         
//         Rcpp::S4 element(Rcpp::as<SEXP>(*it));
//         SEXP address = element.slot("address");
//         
//         // Rcpp::XPtr<dynVCLMat<T> > ptrM(*it->slot("address"));
//         
//         Rcpp::XPtr<dynVCLMat<T> > ptrM(address);
//             
//         int nr = ptrM->nrow();
//         end = nr;
//         
//         // loop through each row in matrix
//         for(int i = 0; i < end; i++){
//             
//             // define range to update
//             viennacl::range r(start, end);
//             
//             // point to range elements
//             viennacl::vector_range<viennacl::vector_base<T> > tmp_r = ptrA->range(r);
//             
//             // get matrix row
//             // assign vector elements with matrix row
//             tmp_r = ptrM->row(i);
//             
//             // update indices
//             start += nr;
//             end += nr;
//         }
//     }
// }


template <typename T>
void
setVCLcols(SEXP ptrA_, CharacterVector names){
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    ptrA->setColumnNames(names);
    return;
}

template <typename T>
StringVector
getVCLcols(SEXP ptrA_){
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    StringVector cnames = ptrA->getColumnNames();
    return cnames;
}

// create identity matrix
template <typename T>
void
cpp_identity_vclMatrix(SEXP ptrA_)
{
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    viennacl::matrix_range<viennacl::matrix<T> > pA = ptrA->data();
    
    // should be square so doesn't matter which dim
    pA = viennacl::identity_matrix<T>(pA.size1());
}


// get diagonal of vclMatrix
template <typename T>
void
cpp_vclMatrix_get_diag(SEXP ptrA_, SEXP ptrB_)
{
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > ptrB(ptrB_);
    
    viennacl::matrix_range<viennacl::matrix<T> > pA = ptrA->data();
    viennacl::vector_range<viennacl::vector_base<T> > pB = ptrB->data();
    
    pB = viennacl::diag(pA);
}


// set diagonal with vclVector
template <typename T>
void
cpp_vclMat_vclVec_set_diag(SEXP ptrA_, SEXP ptrB_)
{
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLVec<T> > ptrB(ptrB_);
    
    viennacl::matrix_range<viennacl::matrix<T> > vcl_A = ptrA->data();
    viennacl::vector_range<viennacl::vector_base<T> > vcl_B = ptrB->data();
    
    viennacl::vector_base<T> diag_A(vcl_A.handle(), std::min(vcl_A.size1(), vcl_A.size2()), 0, vcl_A.internal_size2() + 1);
    
    diag_A = vcl_B;
    
    // vcl_B = viennacl::diag(pA);
}


//copy an existing Xptr
template <typename T>
SEXP
cpp_deepcopy_vclMatrix(SEXP ptrA_, const int ctx_id, const bool source)
{        
    
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    dynVCLMat<T> *mat;
    
    if(source){
        mat = new dynVCLMat<T>(*ptrA->sharedPtr(), ctx_id);
    }else{
        mat = new dynVCLMat<T>(ptrA->data(), ctx_id);
    }
     
    //dynVCLMat<T> *mat = new dynVCLMat<T>(pA, ctx_id);

    Rcpp::XPtr<dynVCLMat<T> > pMat(mat);
    return pMat;
}

//copy an existing Xptr
template <typename T>
SEXP
cpp_deepcopy_vclVector(SEXP ptrA_, int ctx_id)
{        
    Rcpp::XPtr<dynVCLVec<T> > ptrA(ptrA_);
    viennacl::vector_range<viennacl::vector_base<T> > pA  = ptrA->data();
    
    dynVCLVec<T> *vec = new dynVCLVec<T>(pA, ctx_id);
//    dynVCLVec<T> *vec = new dynVCLVec<T>();
//    vec->setVector(pA);
//    vec->setRange(1, pA.size());
//    vec->updateSize();
    
    Rcpp::XPtr<dynVCLVec<T> > pVec(vec);
    return pVec;
}

// slice vclVector
template <typename T>
SEXP
cpp_vclVector_slice(SEXP ptrA_, int start, int end)
{
    Rcpp::XPtr<dynVCLVec<T> > pVec(ptrA_);
    
    dynVCLVec<T> *vec = new dynVCLVec<T>();
    vec->setPtr(pVec->getPtr());
    vec->setRange(start, end);
    vec->updateSize();
    
    XPtr<dynVCLVec<T> > pOut(vec);
    return pOut;
}

//cbind two vclMatrix objects
template <typename T>
SEXP
cpp_cbind_vclMatrix(SEXP ptrA_, SEXP ptrB_, int ctx_id)
{        
    viennacl::context ctx;
    
    // explicitly pull context for thread safe forking
    ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
    
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    viennacl::matrix_range<viennacl::matrix<T> > pA  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > pB  = ptrB->data();
    
    viennacl::matrix<T> C(pA.size1(), pA.size2() + pB.size2(), ctx);
    
    viennacl::matrix_range<viennacl::matrix<T> > C_right(C, viennacl::range(0, pA.size1()), viennacl::range(pA.size2(), pA.size2() + pB.size2()));
    viennacl::matrix_range<viennacl::matrix<T> > C_left(C, viennacl::range(0, pA.size1()), viennacl::range(0, pA.size2()));
    
    C_right = pB;
    C_left = pA;
    
    dynVCLMat<T> *mat = new dynVCLMat<T>(pA.size1(), pA.size2() + pB.size2(), ctx_id);
    mat->setMatrix(C);
    mat->setDims(pA.size1(), pA.size2() + pB.size2());
    mat->setRange(0, pA.size1(), 0, pA.size2() + pB.size2());
    
    Rcpp::XPtr<dynVCLMat<T> > pMat(mat);
    return pMat;
}

//rbind two vclMatrix objects
template <typename T>
SEXP
cpp_rbind_vclMatrix(SEXP ptrA_, SEXP ptrB_, int ctx_id)
{        
    Rcpp::XPtr<dynVCLMat<T> > ptrA(ptrA_);
    Rcpp::XPtr<dynVCLMat<T> > ptrB(ptrB_);
    viennacl::matrix_range<viennacl::matrix<T> > pA  = ptrA->data();
    viennacl::matrix_range<viennacl::matrix<T> > pB  = ptrB->data();
    viennacl::context ctx;
    
    // explicitly pull context for thread safe forking
    ctx = viennacl::context(viennacl::ocl::get_context(static_cast<long>(ctx_id)));
    
    viennacl::matrix<T> C(pA.size1() + pB.size1(), pA.size2(), ctx);
    
    viennacl::matrix_range<viennacl::matrix<T> > C_top(C, viennacl::range(0, pA.size1()), viennacl::range(0, pA.size2()));
    viennacl::matrix_range<viennacl::matrix<T> > C_bottom(C, viennacl::range(pA.size1(), pA.size1() + pB.size1()), viennacl::range(0, pA.size2()));
    
    C_top = pA;
    C_bottom = pB;
    
    dynVCLMat<T> *mat = new dynVCLMat<T>(pA.size1() + pB.size1(), pA.size2(), ctx_id);
    mat->setMatrix(C);
    mat->setDims(pA.size1() + pB.size1(), pA.size2());
    mat->setRange(0, pA.size1() + pB.size1(), 0, pA.size2());
    
    Rcpp::XPtr<dynVCLMat<T> > pMat(mat);
    return pMat;
}

template <typename T>
SEXP
cpp_vclMatrix_block(
    const SEXP ptrA, 
    int rowStart, int rowEnd,
    int colStart, int colEnd)
{
    XPtr<dynVCLMat<T> > pA(ptrA);
    dynVCLMat<T> *mat = new dynVCLMat<T>();
    // mat->setPtr(pA->getPtr());
    mat->setSharedPtr(pA->sharedPtr());
    mat->setRange(rowStart, rowEnd, colStart, colEnd);
    mat->setDims(pA->nrow(), pA->ncol());
    
    XPtr<dynVCLMat<T> > pMat(mat);
    return pMat;
}

// convert XPtr ViennaCL Vector to Eigen vector
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> 
VCLtoVecSEXP(SEXP A_)
{   
    Rcpp::XPtr<dynVCLVec<T> > ptrA(A_);
    
    viennacl::vector_range<viennacl::vector_base<T> > tempA = ptrA->data();

    viennacl::vector_base<T> pA = static_cast<viennacl::vector_base<T> >(tempA);
    int M = pA.size();
    
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am(M);
    
    // viennacl::copy(pA, Am); 
    viennacl::fast_copy(pA.begin(), pA.end(), &(Am[0]));
    
    return Am;
}

// scalar initialized ViennaCL matrix
template <typename T>
SEXP 
cpp_scalar_vclMatrix(
    SEXP scalar_, 
    int nr, 
    int nc,
    int ctx_id)
{
    const T scalar = as<T>(scalar_);
    
    dynVCLMat<T> *mat = new dynVCLMat<T>(nr, nc, scalar, ctx_id);
    Rcpp::XPtr<dynVCLMat<T> > pMat(mat);
    return pMat;
}

// empty ViennaCL matrix
template <typename T>
SEXP cpp_zero_vclMatrix(int nr, int nc, int ctx_id)
{
    dynVCLMat<T> *mat = new dynVCLMat<T>(nr, nc, ctx_id);
    Rcpp::XPtr<dynVCLMat<T> > pMat(mat);
    return pMat;
}

// convert SEXP Vector to ViennaCL vector
template <typename T>
SEXP 
sexpVecToVCL(
    SEXP A,
    int ctx_id)
{        
    dynVCLVec<T> *vec = new dynVCLVec<T>(A, ctx_id);
    Rcpp::XPtr<dynVCLVec<T> > pVec(vec);
    return pVec;
}

// convert SEXP Matrix to ViennaCL matrix
template <typename T>
SEXP cpp_sexp_mat_to_vclMatrix(
    SEXP A, 
    int ctx_id)
{
    dynVCLMat<T> *mat = new dynVCLMat<T>(A, ctx_id);
    Rcpp::XPtr<dynVCLMat<T> > pMat(mat);
    return pMat;
}


// convert XPtr ViennaCL Matrix to Eigen matrix
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> 
VCLtoSEXP(SEXP A)
{
    Rcpp::XPtr<dynVCLMat<T> > ptrA(A);
    viennacl::matrix_range<viennacl::matrix<T> > tempA  = ptrA->data();
    
//    Rcpp::XPtr<viennacl::matrix<T> > pA(A);
//    
//    int nr = pA->size1();
//    int nc = pA->size2();
    
    viennacl::matrix<T> pA = static_cast<viennacl::matrix<T> >(tempA);
    int nr = pA.size1();
    int nc = pA.size2();
    
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Am(nr, nc);
    
    viennacl::copy(pA, Am); 
    
    return Am;
}

// convert SEXP Vector to ViennaCL matrix
template <typename T>
SEXP 
vectorToMatVCL(SEXP A, int nr, int nc, int ctx_id)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Am;
    Am = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(A);
    Am.resize(nr, nc);
    
    //std::cout << "initizlied eigen matrix" << std::endl;

    dynVCLMat<T> *mat = new dynVCLMat<T>(Am, nr, nc, ctx_id);

    //std::cout << "initialized vcl matrix" << std::endl;

    Rcpp::XPtr<dynVCLMat<T> > pMat(mat);
    return pMat;    
    
//    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > Amm(&Am(0), nr, nc);
//    
//    viennacl::matrix<T> *vcl_A = new viennacl::matrix<T>(nr, nc);
//    
//    viennacl::copy(Amm, *vcl_A); 
//    
//    Rcpp::XPtr<viennacl::matrix<T> > pMat(vcl_A);
//    return pMat;
}

// convert ViennaCL matrix to ViennaCL vector (shared memory)
template <typename T>
SEXP
vclMatTovclVec(SEXP ptrA_){
    
    XPtr<dynVCLMat<T> > ptrA(ptrA_);
    viennacl::matrix<T>* ptr = ptrA->getPtr();
    
    // std::cout << ptr << std::endl;
    
    // viennacl::vector_base<T> v(ptr->handle(), ptr->internal_size(), 0, 1);
    
    // std::cout << "outside ptr" << std::endl;
    // std::cout << &v << std::endl;
    
    // std::cout << "initialized vector base" << std::endl;
    // dynVCLVec<T> *vec = new dynVCLVec<T>();
    // vec->setPtr(&v);
    // vec->setRange(0, v.internal_size());
    // vec->updateSize();
    
    // dynVCLVec<T> *vec = new dynVCLVec<T>(static_cast<viennacl::vector<T> >(v));
    dynVCLVec<T> *vec = new dynVCLVec<T>(ptr);
    
    // std::cout << "initialized vcl vector" << std::endl;
    
    Rcpp::XPtr<dynVCLVec<T> > pVec(vec);
    
    // std::cout << "xptr" << std::endl;
    
    return pVec;    
}

// empty ViennaCL Vector
template <typename T>
SEXP emptyVecVCL(
    int length,
    int ctx_id)
{
    dynVCLVec<T> *vec = new dynVCLVec<T>(length, ctx_id);
    Rcpp::XPtr<dynVCLVec<T> > pVec(vec);
    return pVec;
}

/*** vclVector get elements ***/

// Get viennacl column elements
template <typename T>
T
vclVecGetElement(SEXP &data, const int &idx)
{
    Rcpp::XPtr<dynVCLVec<T> > pVec(data);
    viennacl::vector_range<viennacl::vector_base<T> > A  = pVec->data();
    
//    Rcpp::XPtr<viennacl::vector<T> > pA(data);
//    viennacl::vector<T> &A = *pA;
    return(A(idx-1));
}

/*** vclVector set elements ***/

// Get viennacl column elements
template <typename T>
void
vclVecSetElement(SEXP &data, SEXP newdata, const int &idx)
{
    Rcpp::XPtr<dynVCLVec<T> > pA(data);
    viennacl::vector_range<viennacl::vector_base<T> > A = pA->data();
//    viennacl::vector<T> &vec = *A;
    A(idx-1) = as<T>(newdata);
    
//    Rcpp::XPtr<viennacl::vector<T> > pA(data);
//    viennacl::vector<T> &A = *pA;
//    A(idx-1) = as<T>(newdata);
}

// update viennacl matrix with R vector
template <typename T>
void
vclSetVector(SEXP data, SEXP newdata, const int ctx_id)
{
    Rcpp::XPtr<dynVCLVec<T> > pMat(data);
    viennacl::vector_range<viennacl::vector_base<T> > A  = pMat->data();
    
    // move new data to device
    dynVCLVec<T> *mat = new dynVCLVec<T>(newdata, ctx_id);
    
    // access new data on device
    viennacl::vector_range<viennacl::vector_base<T> > A_new = mat->data();
    
    // assign existing matrix with new data
    A = A_new;
}

// update viennacl matrix with another viennacl vector
template <typename T>
void
vclSetVCLVector(SEXP data, SEXP newdata)
{
    Rcpp::XPtr<dynVCLVec<T> > pMat(data);
    Rcpp::XPtr<dynVCLVec<T> > pMatNew(newdata);
    viennacl::vector_range<viennacl::vector_base<T> > A  = pMat->data();
    viennacl::vector_range<viennacl::vector_base<T> > A_new  = pMatNew->data();
    
    // assign existing matrix with new data
    A = A_new;
}

/*** vclMatrix setting elements ***/

// update viennacl column elements
template <typename T>
void
vclSetCol(SEXP data, SEXP newdata, const int nc)
{
    Rcpp::XPtr<dynVCLMat<T> > pMat(data);
    viennacl::matrix_range<viennacl::matrix<T> > A  = pMat->data();
    
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am;
    Am = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(newdata);
    
    for(unsigned int i = 0; i < A.size1(); i++){
        A(i, nc-1) = Am(i);
    } 
    
//    Rcpp::XPtr<viennacl::matrix<T> > pA(data);
//    viennacl::matrix<T> &A = *pA;
//    Eigen::Matrix<T, Eigen::Dynamic, 1> Am;
//    Am = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(newdata);
//    
//    for(unsigned int i = 0; i < A.size1(); i++){
//        A(i, nc-1) = Am(i);
//    } 
}

// update viennacl row elements
template <typename T>
void
vclSetRow(SEXP data, SEXP newdata, const int nr)
{
    Rcpp::XPtr<dynVCLMat<T> > pMat(data);
    viennacl::matrix_range<viennacl::matrix<T> > A  = pMat->data();
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am;
    Am = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(newdata);
    
    for(unsigned int i = 0; i < A.size2(); i++){
        A(nr-1, i) = Am(i);
    } 
    
//    Rcpp::XPtr<viennacl::matrix<T> > pA(data);
//    viennacl::matrix<T> &A = *pA;
//    Eigen::Matrix<T, Eigen::Dynamic, 1> Am;
//    Am = Rcpp::as<Eigen::Matrix<T, Eigen::Dynamic, 1> >(newdata);
//    
//    for(unsigned int i = 0; i < A.size2(); i++){
//        A(nr-1, i) = Am(i);
//    } 
}

// update viennacl element
template <typename T>
void
vclSetElement(SEXP data, SEXP newdata, const int nr, const int nc)
{
    Rcpp::XPtr<dynVCLMat<T> > pMat(data);
    viennacl::matrix_range<viennacl::matrix<T> > A  = pMat->data();
    
//    Rcpp::XPtr<viennacl::matrix<T> > pA(data);
//    viennacl::matrix<T> &A = *pA;
    
    A(nr-1, nc-1) = as<T>(newdata);
}

// update viennacl matrix with R matrix
template <typename T>
void
vclSetMatrix(SEXP data, SEXP newdata, const int ctx_id)
{
    Rcpp::XPtr<dynVCLMat<T> > pMat(data);
    viennacl::matrix_range<viennacl::matrix<T> > A  = pMat->data();
    
    // move new data to device
    dynVCLMat<T> *mat = new dynVCLMat<T>(newdata, ctx_id);
    
    // access new data on device
    viennacl::matrix_range<viennacl::matrix<T> > A_new = mat->data();
    
    // assign existing matrix with new data
    A = A_new;
}

// update viennacl matrix with another viennacl matrix
template <typename T>
void
vclSetVCLMatrix(SEXP data, SEXP newdata, const int ctx_id)
{
    Rcpp::XPtr<dynVCLMat<T> > pMat(data);
    Rcpp::XPtr<dynVCLMat<T> > pMatNew(newdata);
    viennacl::matrix_range<viennacl::matrix<T> > A  = pMat->data();
    viennacl::matrix_range<viennacl::matrix<T> > A_new  = pMatNew->data();
    
    // assign existing matrix with new data
    A = A_new;
}

/*** vclMatrix get elements ***/

// Get viennacl column elements
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
vclGetCol(
    SEXP &data, 
    const int &nc,
    int ctx_id)
{
    Rcpp::XPtr<dynVCLMat<T> > pMat(data);
    viennacl::matrix_range<viennacl::matrix<T> > pA  = pMat->data();
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am;
    Am = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(pA.size1());
    
    viennacl::vector_base<T> vcl_A(pA.size1(), ctx=ctx);
    vcl_A = viennacl::column(pA, nc-1);
    
    // copy(static_cast<viennacl::vector<T> >(vcl_A), Am);
    viennacl::fast_copy(vcl_A.begin(), vcl_A.end(), &(Am[0]));
    return(Am);
}

// Get viennacl row elements
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
vclGetRow(
    SEXP &data, 
    const int &nr,
    int ctx_id)
{
    
    Rcpp::XPtr<dynVCLMat<T> > pMat(data);
    viennacl::matrix_range<viennacl::matrix<T> > pA  = pMat->data();
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
//    Rcpp::XPtr<viennacl::matrix<T> > pA(data);
    Eigen::Matrix<T, Eigen::Dynamic, 1> Am;
    Am = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(pA.size2());
    
    viennacl::vector_base<T> vcl_A(pA.size2(), ctx=ctx);
    vcl_A = viennacl::row(pA, nr-1);
    
    // copy(static_cast<viennacl::vector<T> >(vcl_A), Am);
    viennacl::fast_copy(vcl_A.begin(), vcl_A.end(), &(Am[0]));
    return(Am);
}

// Get viennacl row elements
template <typename T>
T
vclGetElement(SEXP &data, const int &nr, const int &nc)
{
    T value;
    
    Rcpp::XPtr<dynVCLMat<T> > pMat(data);
    viennacl::matrix_range<viennacl::matrix<T> > A  = pMat->data();
    
//    Rcpp::XPtr<viennacl::matrix<T> > pA(data);
//    viennacl::matrix<T> &A = *pA;
    value = A(nr-1, nc-1);
    return(value);
}


// vclMatrix identity matrix
// [[Rcpp::export]]
void
cpp_identity_vclMatrix(SEXP ptrA, const int type_flag)
{
    switch(type_flag) {
        case 4:
            cpp_identity_vclMatrix<int>(ptrA);
            return;
        case 6:
            cpp_identity_vclMatrix<float>(ptrA);
            return;
        case 8:
            cpp_identity_vclMatrix<double>(ptrA);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


/*** vclMatrix diag ***/
// [[Rcpp::export]]
void
cpp_vclMatrix_get_diag(SEXP ptrA, SEXP ptrB, const int type_flag)
{
    switch(type_flag) {
        case 4:
            cpp_vclMatrix_get_diag<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclMatrix_get_diag<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclMatrix_get_diag<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


// vclMatrix set diag with vclVector
// [[Rcpp::export]]
void
cpp_vclMat_vclVec_set_diag(SEXP ptrA, SEXP ptrB, const int type_flag)
{
    switch(type_flag) {
        case 4:
            cpp_vclMat_vclVec_set_diag<int>(ptrA, ptrB);
            return;
        case 6:
            cpp_vclMat_vclVec_set_diag<float>(ptrA, ptrB);
            return;
        case 8:
            cpp_vclMat_vclVec_set_diag<double>(ptrA, ptrB);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


/*** vclMatrix deepcopy ***/
// [[Rcpp::export]]
SEXP
cpp_deepcopy_vclMatrix(SEXP ptrA, 
                       const int type_flag, 
                       const int ctx_id, 
                       const bool source)
{
    switch(type_flag) {
        case 4:
            return cpp_deepcopy_vclMatrix<int>(ptrA, ctx_id, source);
        case 6:
            return cpp_deepcopy_vclMatrix<float>(ptrA, ctx_id, source);
        case 8:
            return cpp_deepcopy_vclMatrix<double>(ptrA, ctx_id, source);
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

/*** vclVector deepcopy ***/
// [[Rcpp::export]]
SEXP
cpp_deepcopy_vclVector(
    SEXP ptrA, 
    const int type_flag,
    int ctx_id)
{
    switch(type_flag) {
        case 4:
            return cpp_deepcopy_vclVector<int>(ptrA, ctx_id);
        case 6:
            return cpp_deepcopy_vclVector<float>(ptrA, ctx_id);
        case 8:
            return cpp_deepcopy_vclVector<double>(ptrA, ctx_id);
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

/*** Slice vclVector ***/
// [[Rcpp::export]]
SEXP
cpp_vclVector_slice(SEXP ptrA, const int start, const int end, const int type_flag)
{    
    switch(type_flag) {
        case 4:
            return cpp_vclVector_slice<int>(ptrA, start, end);
        case 6:
            return cpp_vclVector_slice<float>(ptrA, start, end);
        case 8:
            return cpp_vclVector_slice<double>(ptrA, start, end);
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

    
/*** vclMatrix block ***/
// [[Rcpp::export]]
SEXP
cpp_vclMatrix_block(
    const SEXP ptrA, 
    int rowStart, int rowEnd,
    int colStart, int colEnd,
    const int type_flag)
{    
    switch(type_flag) {
        case 4:
            return cpp_vclMatrix_block<int>(ptrA, rowStart, rowEnd, colStart, colEnd);
        case 6:
            return cpp_vclMatrix_block<float>(ptrA, rowStart, rowEnd, colStart, colEnd);
        case 8:
            return cpp_vclMatrix_block<double>(ptrA, rowStart, rowEnd, colStart, colEnd);
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

/*** vclMatrix cbind ***/
// [[Rcpp::export]]
SEXP
cpp_cbind_vclMatrix(
    SEXP ptrA, 
    SEXP ptrB,
    int type_flag,
    int ctx_id)
{    
    switch(type_flag) {
        case 4:
            return cpp_cbind_vclMatrix<int>(ptrA, ptrB, ctx_id);
        case 6:
            return cpp_cbind_vclMatrix<float>(ptrA, ptrB, ctx_id);
        case 8:
            return cpp_cbind_vclMatrix<double>(ptrA, ptrB, ctx_id);
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

/*** vclMatrix rbind ***/
// [[Rcpp::export]]
SEXP
cpp_rbind_vclMatrix(
    SEXP ptrA, 
    SEXP ptrB,
    int type_flag,
    int ctx_id)
{    
    switch(type_flag) {
        case 4:
            return cpp_rbind_vclMatrix<int>(ptrA, ptrB, ctx_id);
        case 6:
            return cpp_rbind_vclMatrix<float>(ptrA, ptrB, ctx_id);
        case 8:
            return cpp_rbind_vclMatrix<double>(ptrA, ptrB, ctx_id);
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

/*** matrix imports ***/

// [[Rcpp::export]]
SEXP
cpp_sexp_mat_to_vclMatrix(
    SEXP ptrA, 
    int type_flag,
    int ctx_id)
{
    switch(type_flag) {
        case 4:
            return cpp_sexp_mat_to_vclMatrix<int>(ptrA, ctx_id);
        case 6:
            return cpp_sexp_mat_to_vclMatrix<float>(ptrA, ctx_id);
        case 8:
            return cpp_sexp_mat_to_vclMatrix<double>(ptrA, ctx_id);
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


/*** Matrix exports ***/

// [[Rcpp::export]]
SEXP
VCLtoMatSEXP(
    SEXP ptrA, 
    int type_flag)
{
    switch(type_flag) {
        case 4:
            return wrap(VCLtoSEXP<int>(ptrA));
        case 6:
            return wrap(VCLtoSEXP<float>(ptrA));
        case 8:
            return wrap(VCLtoSEXP<double>(ptrA));
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

/*** Matrix initializers ***/

// [[Rcpp::export]]
SEXP
cpp_zero_vclMatrix(
    int nr, 
    int nc, 
    int type_flag,
    int ctx_id)
{
    switch(type_flag) {
        case 4:
            return cpp_zero_vclMatrix<int>(nr, nc, ctx_id);
        case 6:
            return cpp_zero_vclMatrix<float>(nr, nc, ctx_id);
        case 8:
            return cpp_zero_vclMatrix<double>(nr, nc, ctx_id);
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


// [[Rcpp::export]]
SEXP
cpp_scalar_vclMatrix(
    SEXP scalar, 
    int nr, int nc, 
    int type_flag,
    int ctx_id)
{
    switch(type_flag) {
        case 4:
            return cpp_scalar_vclMatrix<int>(scalar, nr, nc, ctx_id);
        case 6:
            return cpp_scalar_vclMatrix<float>(scalar, nr, nc, ctx_id);
        case 8:
            return cpp_scalar_vclMatrix<double>(scalar, nr, nc, ctx_id);
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

/*** matrix element updates ***/

// [[Rcpp::export]]
void
vclSetCol(SEXP ptrA, const int nc, SEXP newdata, const int type_flag)
{
    switch(type_flag) {
        case 4:
            vclSetCol<int>(ptrA, newdata, nc);
            return;
        case 6:
            vclSetCol<float>(ptrA, newdata, nc);
            return;
        case 8:
            vclSetCol<double>(ptrA, newdata, nc);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
void
vclSetRow(SEXP ptrA, const int nr, SEXP newdata, const int type_flag)
{
    switch(type_flag) {
        case 4:
            vclSetRow<int>(ptrA, newdata, nr);
            return;
        case 6:
            vclSetRow<float>(ptrA, newdata, nr);
            return;
        case 8:
            vclSetRow<double>(ptrA, newdata, nr);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
void
vclSetElement(SEXP ptrA, const int nr, const int nc, SEXP newdata, const int type_flag)
{
    switch(type_flag) {
        case 4:
            vclSetElement<int>(ptrA, newdata, nr, nc);
            return;
        case 6:
            vclSetElement<float>(ptrA, newdata, nr, nc);
            return;
        case 8:
            vclSetElement<double>(ptrA, newdata, nr, nc);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
void
vclSetMatrix(SEXP ptrA, SEXP newdata, const int type_flag, const int ctx_id)
{
    switch(type_flag) {
    case 4:
        vclSetMatrix<int>(ptrA, newdata, ctx_id);
        return;
    case 6:
        vclSetMatrix<float>(ptrA, newdata, ctx_id);
        return;
    case 8:
        vclSetMatrix<double>(ptrA, newdata, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
void
vclSetVCLMatrix(SEXP ptrA, SEXP newdata, const int type_flag, const int ctx_id)
{
    switch(type_flag) {
    case 4:
        vclSetVCLMatrix<int>(ptrA, newdata, ctx_id);
        return;
    case 6:
        vclSetVCLMatrix<float>(ptrA, newdata, ctx_id);
        return;
    case 8:
        vclSetVCLMatrix<double>(ptrA, newdata, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

/*** get matrix elements ***/

// [[Rcpp::export]]
SEXP
vclGetCol(SEXP ptrA, const int nc, const int type_flag, int ctx_id)
{
    switch(type_flag) {
        case 4:
            return wrap(vclGetCol<int>(ptrA, nc, ctx_id));
        case 6:
            return wrap(vclGetCol<float>(ptrA, nc, ctx_id));
        case 8:
            return wrap(vclGetCol<double>(ptrA, nc, ctx_id));
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
SEXP
vclGetRow(SEXP ptrA, const int nr, const int type_flag, int ctx_id)
{
    switch(type_flag) {
        case 4:
            return wrap(vclGetRow<int>(ptrA, nr, ctx_id));
        case 6:
            return wrap(vclGetRow<float>(ptrA, nr, ctx_id));
        case 8:
            return wrap(vclGetRow<double>(ptrA, nr, ctx_id));
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
SEXP
vclGetElement(SEXP ptrA, const int nr, const int nc, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return wrap(vclGetElement<int>(ptrA, nr, nc));
        case 6:
            return wrap(vclGetElement<float>(ptrA, nr, nc));
        case 8:
            return wrap(vclGetElement<double>(ptrA, nr, nc));
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

/*** get vector elements ***/

// [[Rcpp::export]]
SEXP
vclVecGetElement(SEXP ptrA, const int idx, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return wrap(vclVecGetElement<int>(ptrA, idx));
        case 6:
            return wrap(vclVecGetElement<float>(ptrA, idx));
        case 8:
            return wrap(vclVecGetElement<double>(ptrA, idx));
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
vclVecSetElement(SEXP ptrA, const int idx, SEXP newdata, const int type_flag)
{
    switch(type_flag) {
        case 4:
            vclVecSetElement<int>(ptrA, newdata, idx);
            return;
        case 6:
            vclVecSetElement<float>(ptrA, newdata, idx);
            return;
        case 8:
            vclVecSetElement<double>(ptrA, newdata, idx);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

// [[Rcpp::export]]
void
vclSetVector(SEXP ptrA, SEXP newdata, const int type_flag, const int ctx_id)
{
    switch(type_flag) {
    case 4:
        vclSetVector<int>(ptrA, newdata, ctx_id);
        return;
    case 6:
        vclSetVector<float>(ptrA, newdata, ctx_id);
        return;
    case 8:
        vclSetVector<double>(ptrA, newdata, ctx_id);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}


// [[Rcpp::export]]
void
vclSetVCLVector(SEXP ptrA, SEXP newdata, const int type_flag)
{
    switch(type_flag) {
    case 4:
        vclSetVCLVector<int>(ptrA, newdata);
        return;
    case 6:
        vclSetVCLVector<float>(ptrA, newdata);
        return;
    case 8:
        vclSetVCLVector<double>(ptrA, newdata);
        return;
    default:
        throw Rcpp::exception("unknown type detected for vclVector object!");
    }
}

/*** vector imports ***/

// [[Rcpp::export]]
SEXP
vectorToVCL(
    SEXP ptrA, 
    int type_flag,
    int ctx_id)
{
    switch(type_flag) {
        case 4:
            return sexpVecToVCL<int>(ptrA, ctx_id);
        case 6:
            return sexpVecToVCL<float>(ptrA, ctx_id);
        case 8:
            return sexpVecToVCL<double>(ptrA, ctx_id);
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
SEXP
vectorToMatVCL(
    SEXP ptrA, 
    int nr,
    int nc,
    int type_flag,
    int ctx_id)
{
    switch(type_flag) {
        case 4:
            return vectorToMatVCL<int>(ptrA, nr, nc, ctx_id);
        case 6:
            return vectorToMatVCL<float>(ptrA, nr, nc, ctx_id);
        case 8:
            return vectorToMatVCL<double>(ptrA, nr, nc, ctx_id);
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


// [[Rcpp::export]]
SEXP
vclMatTovclVec(
    SEXP ptrA, 
    int type_flag)
{
    switch(type_flag) {
    case 4:
        return vclMatTovclVec<int>(ptrA);
    case 6:
        return vclMatTovclVec<float>(ptrA);
    case 8:
        return vclMatTovclVec<double>(ptrA);
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


/*** Vector exports ***/

// [[Rcpp::export]]
SEXP
VCLtoVecSEXP(SEXP ptrA, const int type_flag)
{
    switch(type_flag) {
        case 4:
            return wrap(VCLtoVecSEXP<int>(ptrA));
        case 6:
            return wrap(VCLtoVecSEXP<float>(ptrA));
        case 8:
            return wrap(VCLtoVecSEXP<double>(ptrA));
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

/*** Empty vector initializers ***/

// [[Rcpp::export]]
SEXP
emptyVecVCL(
    int length, 
    int type_flag,
    int ctx_id)
{
    switch(type_flag) {
        case 4:
            return emptyVecVCL<int>(length, ctx_id);
        case 6:
            return emptyVecVCL<float>(length, ctx_id);
        case 8:
            return emptyVecVCL<double>(length, ctx_id);
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}

// [[Rcpp::export]]
void
setVCLcols(SEXP ptrA, CharacterVector names, const int type_flag)
{
    switch(type_flag) {
        case 4:
            setVCLcols<int>(ptrA, names);
            return;
        case 6:
            setVCLcols<float>(ptrA, names);
            return;
        case 8:
            setVCLcols<double>(ptrA, names);
            return;
        default:
            throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


// [[Rcpp::export]]
StringVector
getVCLcols(SEXP ptrA, const int type_flag)
{
    switch(type_flag) {
    case 4:
        return getVCLcols<int>(ptrA);
    case 6:
        return getVCLcols<float>(ptrA);
    case 8:
        return getVCLcols<double>(ptrA);
    default:
        throw Rcpp::exception("unknown type detected for vclMatrix object!");
    }
}


// // [[Rcpp::export]]
// void
// vclMatTovclVec(SEXP ptrA, List ptrList, const int type_flag)
// {
//     switch(type_flag) {
//     case 4:
//         vclMatTovclVec<int>(ptrA, ptrList);
//         return;
//     case 6:
//         vclMatTovclVec<int>(ptrA, ptrList);
//         return;
//     case 8:
//         vclMatTovclVec<int>(ptrA, ptrList);
//         return;
//     default:
//         throw Rcpp::exception("unknown type detected for vclMatrix object");
//     }
// }
