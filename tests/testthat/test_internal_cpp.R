library(gpuR)
context("Check Internal C++ Errors")

# set seed
set.seed(123)

ORDER <- 4

mat <- matrix(rnorm(ORDER ^ 2), ORDER, ORDER)
mat_out <- matrix(0, ORDER, ORDER)
vec <- rnorm(ORDER)
vec_out <- rep(0, ORDER)

gpuA <- gpuMatrix(mat)
gpuC <- gpuMatrix(mat_out)
vclA <- vclMatrix(mat)
vclC <- vclMatrix(mat_out)

gpuV <- gpuVector(vec)
gpuVc <- gpuVector(vec_out)
vclV <- vclVector(vec)
vclVc <- vclVector(vec_out)


test_that("Check gpuMatrix errors thrown when non-implemented type passed", {
    expect_error(
        cpp_gpuMatrix_gemm(gpuA@address, 
                           gpuA@address, 
                           gpuC@address,
                           10L,
                           0L), 
        info = "gemm didn't throw error")
    expect_error(
        cpp_gpuMatrix_crossprod(gpuA@address, 
                                gpuA@address, 
                                gpuC@address,
                                10L,
                                0L), 
        info = "crossprod didn't throw error")
    expect_error(
        cpp_gpuMatrix_tcrossprod(gpuA@address, 
                                 gpuA@address, 
                                 gpuC@address,
                                 10L,
                                 0L), 
        info = "tcrossprod didn't throw error")
    expect_error(
        cpp_gpuMatrix_transpose(gpuA@address, 
                                gpuC@address,
                                10L,
                                0L), 
        info = "transpose didn't throw error")
    expect_error(
        cpp_gpuMatrix_elem_prod(gpuA@address, 
                                gpuA@address, 
                                gpuC@address,
                                10L,
                                0L), 
        info = "elem prod didn't throw error")
    expect_error(
        cpp_gpuMatrix_scalar_prod(gpuA@address, 
                                2, 
                                10L,
                                0L), 
        info = "scalar prod didn't throw error")
    expect_error(
        cpp_gpuMatrix_scalar_div(gpuA@address, 
                                  2, 
                                  10L,
                                  0L), 
        info = "scalar div didn't throw error")
    expect_error(
        cpp_gpuMatrix_elem_div(gpuA@address, 
                                 2, 
                               10L,
                               0L),
        info = "elem div didn't throw error")
    
    expect_error(
        cpp_gpuMatrix_scalar_pow(gpuA@address, 
                                  2, 
                                  10L,
                                  0L), 
        info = "scalar pow didn't throw error")
    
    expect_error(
        cpp_gpuMatrix_elem_pow(gpuA@address, 
                               2, 
                               10L,
                               0L), 
        info = "elem pow didn't throw error")
    
    expect_error(
        cpp_gpuMatrix_elem_sin(gpuA@address, 
                               gpuC@address, 
                               10L,
                               0L), 
        info = "elem sin didn't throw error")
    expect_error(
        cpp_gpuMatrix_elem_asin(gpuA@address, 
                               gpuC@address, 
                               10L,
                               0L), 
        info = "elem asin didn't throw error")
    expect_error(
        cpp_gpuMatrix_elem_sinh(gpuA@address, 
                               gpuC@address, 
                               10L,
                               0L), 
        info = "elem sinh didn't throw error")
    expect_error(
        cpp_gpuMatrix_elem_cos(gpuA@address, 
                               gpuC@address, 
                               10L,
                               0L), 
        info = "elem cos didn't throw error")
    expect_error(
        cpp_gpuMatrix_elem_acos(gpuA@address, 
                               gpuC@address, 
                               10L,
                               0L), 
        info = "elem acos didn't throw error")
    expect_error(
        cpp_gpuMatrix_elem_cosh(gpuA@address, 
                               gpuC@address, 
                               10L,
                               0L), 
        info = "elem cosh didn't throw error")
    expect_error(
        cpp_gpuMatrix_elem_tan(gpuA@address, 
                               gpuC@address, 
                               10L,
                               0L), 
        info = "elem tan didn't throw error")
    expect_error(
        cpp_gpuMatrix_elem_atan(gpuA@address, 
                               gpuC@address, 
                               10L,
                               0L), 
        info = "elem atan didn't throw error")
    expect_error(
        cpp_gpuMatrix_elem_tanh(gpuA@address, 
                               gpuC@address, 
                               10L,
                               0L), 
        info = "elem tanh didn't throw error")
    expect_error(
        cpp_gpuMatrix_elem_log(gpuA@address, 
                               gpuC@address, 
                               10L,
                               0L), 
        info = "elem log didn't throw error")
    expect_error(
        cpp_gpuMatrix_elem_log_base(gpuA@address, 
                               gpuC@address, 
                               2,
                               10L,
                               0L), 
        info = "elem log_base didn't throw error")
    expect_error(
        cpp_gpuMatrix_elem_log10(gpuA@address, 
                               gpuC@address, 
                               10L,
                               0L), 
        info = "elem log10 didn't throw error")
    expect_error(
        cpp_gpuMatrix_elem_exp(gpuA@address, 
                               gpuC@address, 
                               10L,
                               0L), 
        info = "elem exp didn't throw error")
    expect_error(
        cpp_gpuMatrix_elem_abs(gpuA@address, 
                               gpuC@address, 
                               10L,
                               0L), 
        info = "elem abs didn't throw error")
    expect_error(
        cpp_gpuMatrix_axpy(2,
                           gpuA@address, 
                           gpuC@address, 
                           10L,
                           0L), 
        info = "AXPY didn't throw error")
    expect_error(
        cpp_gpuMatrix_axpy(gpuA@address, 
                           10L,
                           0L), 
        info = "unary AXPY didn't throw error")
    expect_error(
        cpp_gpuMatrix_max(gpuA@address, 
                          10L,
                          0L), 
        info = "max didn't throw error")
    expect_error(
        cpp_gpuMatrix_min(gpuA@address, 
                          10L,
                          0L), 
        info = "min didn't throw error")
    
    expect_error(
        cpp_gpu_eigen(gpuA@address, 
                      gpuA@address, 
                      gpuV@address,
                      TRUE,
                      10L,
                      0L),
        info = "eigen didn't throw error"
    )
    expect_error(
        cpp_gpuMatrix_colmean(gpuA@address, 
                              gpuV@address,
                              10L,
                              0L),
        info = "colmean didn't throw error"
    )
    expect_error(
        cpp_gpuMatrix_colsum(gpuA@address, 
                              gpuV@address,
                              10L,
                              0L),
        info = "colsum didn't throw error"
    )
    expect_error(
        cpp_gpuMatrix_rowmean(gpuA@address, 
                              gpuV@address,
                              10L,
                              0L),
        info = "rowmean didn't throw error"
    )
    expect_error(
        cpp_gpuMatrix_rowsums(gpuA@address, 
                              gpuV@address,
                              10L,
                              0L),
        info = "rowsums didn't throw error"
    )
    
    expect_error(
        cpp_gpuMatrix_pmcc(gpuA@address, 
                           gpuA@address,
                           10L,
                           0L),
        info = "pmcc didn't throw error"
    )
    expect_error(
        cpp_gpuMatrix_eucl(gpuA@address, 
                           gpuA@address,
                           TRUE,
                           10L,
                           0L),
        info = "eucl didn't throw error"
    )
    expect_error(
        cpp_gpuMatrix_peucl(gpuA@address, 
                            gpuA@address,
                            gpuC@address,
                            TRUE,
                            10L,
                            0L),
        info = "eucl didn't throw error"
    )
})

test_that("Check vclMatrix errors thrown when non-implemented type passed", {
    
    expect_error(
        cpp_vclMatrix_gemm(vclA@address, 
                           vclA@address, 
                           vclC@address,
                           10L), 
        info = "gemm didn't throw error")
    expect_error(
        cpp_vclMatrix_crossprod(vclA@address, 
                                vclA@address, 
                                vclC@address,
                                10L), 
        info = "crossprod didn't throw error")
    expect_error(
        cpp_vclMatrix_tcrossprod(vclA@address, 
                                 vclA@address, 
                                 vclC@address,
                                 10L), 
        info = "tcrossprod didn't throw error")
    expect_error(
        cpp_vclMatrix_transpose(vclA@address, 
                                vclC@address,
                                10L), 
        info = "transpose didn't throw error")
    expect_error(
        cpp_vclMatrix_elem_prod(vclA@address, 
                                vclA@address, 
                                vclC@address,
                                10L), 
        info = "elem prod didn't throw error")
    expect_error(
        cpp_vclMatrix_scalar_prod(vclA@address, 
                                  2, 
                                  10L), 
        info = "scalar prod didn't throw error")
    expect_error(
        cpp_vclMatrix_scalar_div(vclA@address, 
                                 2, 
                                 10L), 
        info = "scalar div didn't throw error")
    expect_error(
        cpp_vclMatrix_elem_div(vclA@address, 
                               2, 
                               10L), 
        info = "elem div didn't throw error")
    
    expect_error(
        cpp_vclMatrix_scalar_pow(vclA@address, 
                                 2, 
                                 10L), 
        info = "scalar pow didn't throw error")
    
    expect_error(
        cpp_vclMatrix_elem_pow(vclA@address, 
                               2, 
                               10L), 
        info = "elem pow didn't throw error")
    
    expect_error(
        cpp_vclMatrix_elem_sin(vclA@address, 
                               vclC@address, 
                               10L), 
        info = "elem sin didn't throw error")
    expect_error(
        cpp_vclMatrix_elem_asin(vclA@address, 
                                vclC@address, 
                                10L), 
        info = "elem asin didn't throw error")
    expect_error(
        cpp_vclMatrix_elem_sinh(vclA@address, 
                                vclC@address, 
                                10L), 
        info = "elem sinh didn't throw error")
    expect_error(
        cpp_vclMatrix_elem_cos(vclA@address, 
                               vclC@address, 
                               10L), 
        info = "elem cos didn't throw error")
    expect_error(
        cpp_vclMatrix_elem_acos(vclA@address, 
                                vclC@address, 
                                10L), 
        info = "elem acos didn't throw error")
    expect_error(
        cpp_vclMatrix_elem_cosh(vclA@address, 
                                vclC@address, 
                                10L), 
        info = "elem cosh didn't throw error")
    expect_error(
        cpp_vclMatrix_elem_tan(vclA@address, 
                               vclC@address, 
                               10L), 
        info = "elem tan didn't throw error")
    expect_error(
        cpp_vclMatrix_elem_atan(vclA@address, 
                                vclC@address, 
                                10L), 
        info = "elem atan didn't throw error")
    expect_error(
        cpp_vclMatrix_elem_tanh(vclA@address, 
                                vclC@address, 
                                10L), 
        info = "elem tanh didn't throw error")
    expect_error(
        cpp_vclMatrix_elem_log(vclA@address, 
                               vclC@address, 
                               10L), 
        info = "elem log didn't throw error")
    expect_error(
        cpp_vclMatrix_elem_log_base(vclA@address, 
                                    vclC@address, 
                                    2,
                                    10L), 
        info = "elem log_base didn't throw error")
    expect_error(
        cpp_vclMatrix_elem_log10(vclA@address, 
                                 vclC@address, 
                                 10L), 
        info = "elem log10 didn't throw error")
    expect_error(
        cpp_vclMatrix_elem_exp(vclA@address, 
                               vclC@address, 
                               10L), 
        info = "elem exp didn't throw error")
    expect_error(
        cpp_vclMatrix_elem_abs(vclA@address, 
                               vclC@address, 
                               10L), 
        info = "elem abs didn't throw error")
    expect_error(
        cpp_vclMatrix_axpy(2,
                           vclA@address, 
                           vclC@address, 
                           10L), 
        info = "AXPY didn't throw error")
    expect_error(
        cpp_vclMatrix_axpy(vclA@address, 
                           10L), 
        info = "unary AXPY didn't throw error")
    expect_error(
        cpp_vclMatrix_max(vclA@address, 
                          10L), 
        info = "max didn't throw error")
    expect_error(
        cpp_vclMatrix_min(vclA@address, 
                          10L), 
        info = "min didn't throw error")
    
    expect_error(
        cpp_vcl_eigen(vclA@address, 
                      vclA@address, 
                      vclV@address,
                      TRUE,
                      10L,
                      0L),
        info = "eigen didn't throw error"
    )
    expect_error(
        cpp_vclMatrix_colmean(vclA@address, 
                              vclV@address,
                              10L,
                              0L),
        info = "colmean didn't throw error"
    )
    expect_error(
        cpp_vclMatrix_colsum(vclA@address, 
                             vclV@address,
                             10L,
                             0L),
        info = "colsum didn't throw error"
    )
    expect_error(
        cpp_vclMatrix_rowmean(vclA@address, 
                              vclV@address,
                              10L,
                              0L),
        info = "rowmean didn't throw error"
    )
    expect_error(
        cpp_vclMatrix_rowsums(vclA@address, 
                              vclV@address,
                              10L,
                              0L),
        info = "rowsums didn't throw error"
    )
    expect_error(
        cpp_vclMatrix_pmcc(vclA@address, 
                           vclA@address,
                           10L,
                           0L),
        info = "pmcc didn't throw error"
    )
    expect_error(
        cpp_vclMatrix_eucl(vclA@address, 
                           vclA@address,
                           TRUE,
                           10L,
                           0L),
        info = "eucl didn't throw error"
    )
    expect_error(
        cpp_vclMatrix_peucl(vclA@address, 
                            vclA@address,
                            vclC@address,
                            TRUE,
                            10L,
                            0L),
        info = "eucl didn't throw error"
    )
})

test_that("Check gpuVector errors thrown when non-implemented type passed", {
    
    expect_error(
        cpp_gpuVector_elem_prod(gpuV@address, 
                                gpuV@address, 
                                gpuVc@address,
                                10L,
                                0L), 
        info = "elem prod didn't throw error")
    expect_error(
        cpp_gpuVector_scalar_prod(gpuV@address, 
                                  2, 
                                  10L,
                                  0L), 
        info = "scalar prod didn't throw error")
    expect_error(
        cpp_gpuVector_scalar_div(gpuV@address, 
                                 2, 
                                 10L,
                                 0L), 
        info = "scalar div didn't throw error")
    expect_error(
        cpp_gpuVector_elem_div(gpuV@address, 
                               2, 
                               10L,
                               0L), 
        info = "elem div didn't throw error")
    
    expect_error(
        cpp_gpuVector_scalar_pow(gpuV@address, 
                                 2, 
                                 10L,
                                 0L), 
        info = "scalar pow didn't throw error")
    
    expect_error(
        cpp_gpuVector_elem_pow(gpuV@address, 
                               2, 
                               10L,
                               0L), 
        info = "elem pow didn't throw error")
    
    expect_error(
        cpp_gpuVector_elem_sin(gpuV@address, 
                               gpuVc@address, 
                               10L,
                               0L), 
        info = "elem sin didn't throw error")
    expect_error(
        cpp_gpuVector_elem_asin(gpuV@address, 
                                gpuVc@address, 
                                10L,
                                0L), 
        info = "elem asin didn't throw error")
    expect_error(
        cpp_gpuVector_elem_sinh(gpuV@address, 
                                gpuVc@address, 
                                10L,
                                0L), 
        info = "elem sinh didn't throw error")
    expect_error(
        cpp_gpuVector_elem_cos(gpuV@address, 
                               gpuVc@address, 
                               10L,
                               0L), 
        info = "elem cos didn't throw error")
    expect_error(
        cpp_gpuVector_elem_acos(gpuV@address, 
                                gpuVc@address, 
                                10L,
                                0L), 
        info = "elem acos didn't throw error")
    expect_error(
        cpp_gpuVector_elem_cosh(gpuV@address, 
                                gpuVc@address, 
                                10L,
                                0L), 
        info = "elem cosh didn't throw error")
    expect_error(
        cpp_gpuVector_elem_tan(gpuV@address, 
                               gpuVc@address, 
                               10L,
                               0L), 
        info = "elem tan didn't throw error")
    expect_error(
        cpp_gpuVector_elem_atan(gpuV@address, 
                                gpuVc@address, 
                                10L,
                                0L), 
        info = "elem atan didn't throw error")
    expect_error(
        cpp_gpuVector_elem_tanh(gpuV@address, 
                                gpuVc@address, 
                                10L,
                                0L), 
        info = "elem tanh didn't throw error")
    expect_error(
        cpp_gpuVector_elem_log(gpuV@address, 
                               gpuVc@address, 
                               10L,
                               0L), 
        info = "elem log didn't throw error")
    expect_error(
        cpp_gpuVector_elem_log_base(gpuV@address, 
                                    gpuVc@address, 
                                    2,
                                    10L,
                                    0L), 
        info = "elem log_base didn't throw error")
    expect_error(
        cpp_gpuVector_elem_log10(gpuV@address, 
                                 gpuVc@address, 
                                 10L,
                                 0L), 
        info = "elem log10 didn't throw error")
    expect_error(
        cpp_gpuVector_elem_exp(gpuV@address, 
                               gpuVc@address, 
                               10L,
                               0L), 
        info = "elem exp didn't throw error")
    expect_error(
        cpp_gpuVector_elem_abs(gpuV@address, 
                               gpuVc@address, 
                               10L,
                               0L), 
        info = "elem abs didn't throw error")
    expect_error(
        cpp_gpuVector_axpy(2,
                           gpuV@address, 
                           gpuVc@address, 
                           10L,
                           0L), 
        info = "AXPY didn't throw error")
    expect_error(
        cpp_gpuVector_axpy(gpuV@address, 
                           10L,
                           0L), 
        info = "unary AXPY didn't throw error")
    expect_error(
        cpp_gpuVector_inner_prod(gpuV@address, 
                                 gpuV@address, 
                                 10L,
                                 0L), 
        info = "inner prod didn't throw error")
    expect_error(
        cpp_gpuVector_outer_prod(gpuV@address, 
                                 gpuV@address, 
                                 gpuC@address,
                                 10L,
                                 0L), 
        info = "outer prod didn't throw error")
    expect_error(
        cpp_gpuVector_max(gpuV@address, 
                          10L,
                          0L), 
        info = "max didn't throw error")
    expect_error(
        cpp_gpuVector_min(gpuV@address, 
                          10L,
                          0L), 
        info = "min didn't throw error")
})

test_that("Check vclVector errors thrown when non-implemented type passed", {
    
    expect_error(
        cpp_vclVector_elem_prod(vclV@address, 
                                vclV@address, 
                                vclVc@address,
                                10L), 
        info = "elem prod didn't throw error")
    expect_error(
        cpp_vclVector_scalar_prod(vclV@address, 
                                  2, 
                                  10L), 
        info = "scalar prod didn't throw error")
    expect_error(
        cpp_vclVector_scalar_div(vclV@address, 
                                 2, 
                                 10L), 
        info = "scalar div didn't throw error")
    expect_error(
        cpp_vclVector_elem_div(vclV@address, 
                               2, 
                               10L), 
        info = "elem div didn't throw error")
    
    expect_error(
        cpp_vclVector_scalar_pow(vclV@address, 
                                 2, 
                                 10L), 
        info = "scalar pow didn't throw error")
    
    expect_error(
        cpp_vclVector_elem_pow(vclV@address, 
                               2, 
                               10L), 
        info = "elem pow didn't throw error")
    
    expect_error(
        cpp_vclVector_elem_sin(vclV@address, 
                               vclVc@address, 
                               10L), 
        info = "elem sin didn't throw error")
    expect_error(
        cpp_vclVector_elem_asin(vclV@address, 
                                vclVc@address, 
                                10L), 
        info = "elem asin didn't throw error")
    expect_error(
        cpp_vclVector_elem_sinh(vclV@address, 
                                vclVc@address, 
                                10L), 
        info = "elem sinh didn't throw error")
    expect_error(
        cpp_vclVector_elem_cos(vclV@address, 
                               vclVc@address, 
                               10L), 
        info = "elem cos didn't throw error")
    expect_error(
        cpp_vclVector_elem_acos(vclV@address, 
                                vclVc@address, 
                                10L), 
        info = "elem acos didn't throw error")
    expect_error(
        cpp_vclVector_elem_cosh(vclV@address, 
                                vclVc@address, 
                                10L), 
        info = "elem cosh didn't throw error")
    expect_error(
        cpp_vclVector_elem_tan(vclV@address, 
                               vclVc@address, 
                               10L), 
        info = "elem tan didn't throw error")
    expect_error(
        cpp_vclVector_elem_atan(vclV@address, 
                                vclVc@address, 
                                10L), 
        info = "elem atan didn't throw error")
    expect_error(
        cpp_vclVector_elem_tanh(vclV@address, 
                                vclVc@address, 
                                10L), 
        info = "elem tanh didn't throw error")
    expect_error(
        cpp_vclVector_elem_log(vclV@address, 
                               vclVc@address, 
                               10L), 
        info = "elem log didn't throw error")
    expect_error(
        cpp_vclVector_elem_log_base(vclV@address, 
                                    vclVc@address, 
                                    2,
                                    10L), 
        info = "elem log_base didn't throw error")
    expect_error(
        cpp_vclVector_elem_log10(vclV@address, 
                                 vclVc@address, 
                                 10L), 
        info = "elem log10 didn't throw error")
    expect_error(
        cpp_vclVector_elem_exp(vclV@address, 
                               vclVc@address, 
                               10L), 
        info = "elem exp didn't throw error")
    expect_error(
        cpp_vclVector_elem_abs(vclV@address, 
                               vclVc@address, 
                               10L), 
        info = "elem abs didn't throw error")
    expect_error(
        cpp_vclVector_axpy(2,
                           vclV@address, 
                           vclVc@address, 
                           10L), 
        info = "AXPY didn't throw error")
    expect_error(
        cpp_vclVector_axpy(vclV@address, 
                           10L), 
        info = "unary AXPY didn't throw error")
    expect_error(
        cpp_vclVector_max(vclV@address, 
                          10L), 
        info = "max didn't throw error")
    expect_error(
        cpp_vclVector_min(vclV@address, 
                          10L), 
        info = "min didn't throw error")
    expect_error(
        cpp_vclVector_inner_prod(vclV@address, 
                                 vclV@address, 
                                 10L), 
        info = "inner prod didn't throw error")
    expect_error(
        cpp_vclVector_outer_prod(vclV@address, 
                                 vclV@address, 
                                 vclC@address,
                                 10L), 
        info = "outer prod didn't throw error")
})

