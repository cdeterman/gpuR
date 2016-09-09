library(gpuR)
context("vclMatrix solve")

# set seed
set.seed(123)

ORDER <- 10

# Base R objects
X <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
nsqA <- matrix(rnorm(20), nrow = 4)

rinv <- solve(X)

test_that("vclMatrix Single Precision Matrix Square Matrix Inversion",
          {
              
              has_gpu_skip()
              
              fgpuX <- vclMatrix(X, type="float")
              fgpuA <- vclMatrix(nsqA, type = "float")
              
              ginv <- solve(fgpuX)
              
              expect_is(ginv, "fvclMatrix")
              
              # make sure X not overwritten
              expect_equal(fgpuX[], X, tolerance = 1e-05,
                           info = "input matrix was overwritten")
              expect_equal(ginv[], rinv, tolerance=1e-05, 
                           info="float matrix inverses not equivalent") 
              expect_error(solve(fgpuA), "non-square matrix not currently supported for 'solve'",
                           info = "solve shouldn't accept non-square matrices")
          })


test_that("vclMatrix Single Precision Matrix second matrix inversion",
          {
              
              has_gpu_skip()
              
              fgpuX <- vclMatrix(X, type="float")
              fgpuA <- vclMatrix(nsqA, type = "float")
              iMat <- identity_matrix(nrow(fgpuX), type = "float")
              
              ginv <- solve(fgpuX, iMat)
              
              expect_is(ginv, "fvclMatrix")
              
              # make sure X not overwritten
              expect_equal(fgpuX[], X, tolerance = 1e-05,
                           info = "input matrix was overwritten")
              expect_equal(ginv[], rinv, tolerance=1e-05, 
                           info="float matrix inverses not equivalent") 
              expect_error(solve(fgpuA), "non-square matrix not currently supported for 'solve'",
                           info = "solve shouldn't accept non-square matrices")
          })


test_that("vclMatrix Double Precision Matrix Square Matrix Inversion", 
          {
              
              has_gpu_skip()
              has_double_skip()
              
              fgpuX <- vclMatrix(X, type="double")
              fgpuA <- vclMatrix(nsqA, type = "double")
              
              ginv <- solve(fgpuX)
              
              expect_is(ginv, "dvclMatrix")
              
              expect_equal(fgpuX[], X, tolerance = .Machine$double.eps ^ 0.5,
                           info = "input matrix was overwritten")
              expect_equal(ginv[], rinv, tolerance=.Machine$double.eps ^ 0.5, 
                           info="double matrix inverses not equivalent") 
              expect_error(solve(fgpuA), "non-square matrix not currently supported for 'solve'",
                           info = "solve shouldn't accept non-square matrices")
          })


test_that("vclMatrix Double Precision Matrix second matrix inversion", 
          {
              
              has_gpu_skip()
              has_double_skip()
              
              fgpuX <- vclMatrix(X, type="double")
              fgpuA <- vclMatrix(nsqA, type = "double")
              iMat <- identity_matrix(nrow(fgpuX), type = "double")
              
              ginv <- solve(fgpuX, iMat)
              
              expect_is(ginv, "dvclMatrix")
              
              expect_equal(fgpuX[], X, tolerance = .Machine$double.eps ^ 0.5,
                           info = "input matrix was overwritten")
              expect_equal(ginv[], rinv, tolerance=.Machine$double.eps ^ 0.5, 
                           info="double matrix inverses not equivalent") 
              expect_error(solve(fgpuA), "non-square matrix not currently supported for 'solve'",
                           info = "solve shouldn't accept non-square matrices")
          })

