library(gpuR)
context("gpuMatrix chol decomposition")

# set seed
set.seed(123)

ORDER <- 10

# Base R objects
X <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
X <- X %*% t(X)
# X <- matrix(c(5,1,1,3),2,2)
# nsqA <- matrix(rnorm(20), nrow = 4)

D <- chol(X)

test_that("gpuMatrix Single Precision Matrix Cholesky Decomposition",
          {
              
              has_gpu_skip()
              
              fgpuX <- gpuMatrix(X, type="float")
              # fgpuA <- gpuMatrix(nsqA, type = "float")
              
              C <- chol(fgpuX)
              
              # print(C[])
              
              expect_is(C, "fgpuMatrix")
              expect_equal(C[], D, tolerance=1e-05, 
                           info="float cholesky decomposition not equivalent")  
              
              
              # expect_error(svd(fgpuA), "non-square matrix not currently supported for 'svd'",
              #              info = "svd shouldn't accept non-square matrices")
          })

test_that("gpuMatrix Double Precision Matrix Cholesky Decomposition",
          {
              
              has_gpu_skip()
              has_double_skip()
              
              fgpuX <- gpuMatrix(X, type="double")
              
              C <- chol(fgpuX)
              
              # print(C[])
              
              expect_is(C, "dgpuMatrix")
              expect_equal(C[], D, tolerance=.Machine$double.eps^0.5, 
                           info="double cholesky decomposition not equivalent")  
              
              
              # expect_error(svd(fgpuA), "non-square matrix not currently supported for 'svd'",
              #              info = "svd shouldn't accept non-square matrices")
          })
