library(gpuR)
context("CPU gpuMatrix qr decomposition")

# set seed
set.seed(123)

ORDER <- 10

# Base R objects
X <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
nsqA <- matrix(rnorm(20), nrow = 4)

qrX <- qr(X)
Q <- qr.Q(qrX)
R <- qr.R(qrX)

test_that("CPU gpuMatrix Single Precision Matrix QR Decomposition",
          {
              
              has_cpu_skip()
              
              fgpuX <- gpuMatrix(X, type="float")
              fgpuA <- gpuMatrix(nsqA, type = "float")
              
              E <- qr(fgpuX)
              
              gpuQ <- qr.Q(E)
              gpuR <- qr.R(E)
              
              expect_is(E, "gpuQR")
              # need abs as some signs are opposite (not important with eigenvectors)
              expect_equal(abs(gpuQ[]), abs(Q), tolerance=1e-05,
                           info="Q matrices not equivalent")
              
              # make sure X not overwritten
              expect_equal(abs(gpuR[]), abs(R), tolerance=1e-05,
                           info="R matrices not equivalent")
              expect_error(qr(fgpuA), "non-square matrix not currently supported for 'qr'",
                           info = "qr shouldn't accept non-square matrices")
          })

test_that("CPU gpuMatrix Double Precision Matrix QR Decomposition",
          {
              
              has_cpu_skip()
              
              fgpuX <- gpuMatrix(X, type="double")
              fgpuA <- gpuMatrix(nsqA, type = "double")
              
              E <- qr(fgpuX)
              gpuQ <- qr.Q(E)
              gpuR <- qr.R(E)
              
              expect_is(E, "gpuQR")
              expect_equal(abs(gpuQ[]), abs(Q), tolerance=.Machine$double.eps ^ 0.5,
                           info="Q matrices not equivalent")
              expect_equal(abs(gpuR[]), abs(R), tolerance=.Machine$double.eps ^ 0.5,
                           info="R matrices not equivalent")
              
              expect_error(qr(fgpuA), "non-square matrix not currently supported for 'qr'",
                           info = "qr shouldn't accept non-square matrices")
          })
