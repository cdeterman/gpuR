library(gpuR)
context("vclMatrix solve")

# set seed
set.seed(123)

ORDER <- 10

# Base R objects
X <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
Y <- matrix(rnorm(ORDER), nrow = ORDER)
nsqA <- matrix(rnorm(20), nrow = 4)
nY <- matrix(rnorm(ORDER - 1), nrow = ORDER - 1)

iX <- matrix(sample(seq.int(16), 16), 4)

rinv <- solve(X)
ninv <- solve(X,Y)

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


test_that("vclMatrix Single Precision Matrix non-identity solve",
          {
              
              has_gpu_skip()
              
              fgpuX <- vclMatrix(X, type="float")
              iMat <- vclMatrix(Y, type = "float")
              nMat <- vclMatrix(nY, type = "float")
              
              ginv <- solve(fgpuX, iMat)
              
              expect_is(ginv, "fvclMatrix")
              
              expect_equal(ginv[], ninv, tolerance=1e-05,
                           info="float matrix inverses not equivalent")
              expect_error(solve(fgpuA, nMat),
                           info = "matrices must be compatible, 
                           should return an error")
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


test_that("vclMatrix Double Precision Matrix non-identity solve",
          {
              
              has_gpu_skip()
              has_double_skip()
              
              fgpuX <- vclMatrix(X, type="double")
              iMat <- vclMatrix(Y, type = "double")
              nMat <- vclMatrix(nY, type = "double")
              
              ginv <- solve(fgpuX, iMat)
              
              expect_is(ginv, "dvclMatrix")
              
              expect_equal(ginv[], ninv, tolerance=.Machine$double.eps^0.5,
                           info="double matrix inverses not equivalent")
              expect_error(solve(fgpuA, nMat),
                           info = "matrices must be compatible, 
                           should return an error")
          })


test_that("vclMatrix Integer Inversion not supported",
          {
              
              has_gpu_skip()
              
              fgpuX <- vclMatrix(X, type="integer")
              iMat <- vclMatrix(diag(nrow(X)), type = "float")
              
              expect_error(solve(fgpuX), "Integer solve not implemented",
                           info = "solve shouldn't accept integer matrices")
              expect_error(solve(fgpuX, iMat), "Integer solve not implemented",
                           info = "solve shouldn't accept integer matrices")
          })
