library(gpuR)
context("gpuMatrix solve")

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

test_that("gpuMatrix Single Precision Matrix Square Matrix Inversion",
          {

              has_gpu_skip()

              fgpuX <- gpuMatrix(X, type="float")
              fgpuA <- gpuMatrix(nsqA, type = "float")

              ginv <- solve(fgpuX)

              # print("completed")

              expect_is(ginv, "fvclMatrix")

              # print("R matrix")
              # print(rinv)
              # print("gpu matrix")
              # print(ginv[])

              # make sure X not overwritten
              expect_equal(fgpuX[], X, tolerance = 1e-05,
                           info = "input matrix was overwritten")
              expect_equal(ginv[], rinv, tolerance=1e-05,
                           info="float matrix inverses not equivalent")
              expect_error(solve(fgpuA), "non-square matrix not currently supported for 'solve'",
                           info = "solve shouldn't accept non-square matrices")
          })


test_that("gpuMatrix Single Precision Matrix second matrix inversion",
          {

              has_gpu_skip()

              fgpuX <- gpuMatrix(X, type="float")
              fgpuA <- gpuMatrix(nsqA, type = "float")
              # iMat <- identity_matrix(nrow(fgpuX), type = "float")
              iMat <- gpuMatrix(diag(nrow(X)), type = "float")

              ginv <- solve(fgpuX, iMat)

              expect_is(ginv, "fgpuMatrix")

              # make sure X not overwritten
              expect_equal(fgpuX[], X, tolerance = 1e-05,
                           info = "input matrix was overwritten")
              expect_equal(ginv[], rinv, tolerance=1e-05,
                           info="float matrix inverses not equivalent")
              expect_error(solve(fgpuA), "non-square matrix not currently supported for 'solve'",
                           info = "solve shouldn't accept non-square matrices")
          })


test_that("gpuMatrix Single Precision Matrix non-identity solve",
          {
              
              has_gpu_skip()
              
              fgpuX <- gpuMatrix(X, type="float")
              iMat <- gpuMatrix(Y, type = "float")
              nMat <- gpuMatrix(nY, type = "float")
              
              ginv <- solve(fgpuX, iMat)
              
              expect_is(ginv, "fgpuMatrix")
              
              expect_equal(ginv[], ninv, tolerance=1e-05,
                           info="float matrix inverses not equivalent")
              expect_error(solve(fgpuA, nMat),
                           info = "matrices must be compatible, 
                           should return an error")
          })


test_that("gpuMatrix Double Precision Matrix Square Matrix Inversion",
          {

              has_gpu_skip()
              has_double_skip()

              fgpuX <- gpuMatrix(X, type="double")
              fgpuA <- gpuMatrix(nsqA, type = "double")

              ginv <- solve(fgpuX)

              expect_is(ginv, "dvclMatrix")

              expect_equal(fgpuX[], X, tolerance = .Machine$double.eps ^ 0.5,
                           info = "input matrix was overwritten")
              expect_equal(ginv[], rinv, tolerance=.Machine$double.eps ^ 0.5,
                           info="double matrix inverses not equivalent")
              expect_error(solve(fgpuA), "non-square matrix not currently supported for 'solve'",
                           info = "solve shouldn't accept non-square matrices")
          })


test_that("gpuMatrix Double Precision Matrix second matrix inversion",
          {

              has_gpu_skip()
              has_double_skip()

              fgpuX <- gpuMatrix(X, type="double")
              fgpuA <- gpuMatrix(nsqA, type = "double")
              iMat <- gpuMatrix(diag(nrow(X)), type = "double")

              ginv <- solve(fgpuX, iMat)

              expect_is(ginv, "dgpuMatrix")

              expect_equal(fgpuX[], X, tolerance = .Machine$double.eps ^ 0.5,
                           info = "input matrix was overwritten")
              expect_equal(ginv[], rinv, tolerance=.Machine$double.eps ^ 0.5,
                           info="double matrix inverses not equivalent")
              expect_error(solve(fgpuA), "non-square matrix not currently supported for 'solve'",
                           info = "solve shouldn't accept non-square matrices")
          })


test_that("gpuMatrix Double Precision Matrix non-identity solve",
          {
              
              has_gpu_skip()
              has_double_skip()
              
              fgpuX <- gpuMatrix(X, type="double")
              iMat <- gpuMatrix(Y, type = "double")
              nMat <- gpuMatrix(nY, type = "double")
              
              ginv <- solve(fgpuX, iMat)
              
              expect_is(ginv, "dgpuMatrix")
              
              expect_equal(ginv[], ninv, tolerance=.Machine$double.eps^0.5,
                           info="double matrix inverses not equivalent")
              expect_error(solve(fgpuA, nMat),
                           info = "matrices must be compatible, 
                           should return an error")
          })


test_that("gpuMatrix Integer Inversion not supported",
          {
              
              has_gpu_skip()
              
              fgpuX <- gpuMatrix(X, type="integer")
              iMat <- gpuMatrix(diag(nrow(X)), type = "float")
              
              expect_error(solve(fgpuX), "Integer solve not implemented",
                           info = "solve shouldn't accept integer matrices")
              expect_error(solve(fgpuX, iMat), "Integer solve not implemented",
                           info = "solve shouldn't accept integer matrices")
          })
