library(gpuR)
context("CPU gpuMatrixBlock algebra")

# set option to use CPU instead of GPU
options(gpuR.default.device.type = "cpu")

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
Aint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
Bint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
B <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
E <- matrix(rnorm(15), nrow=5)

# Single Precision Matrix Block tests

test_that("CPU gpuMatrix Single Precision Block Matrix multiplication", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS %*% BS
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    
    fgpuAS <- block(fgpuA, 2L,4L,2L,4L)
    fgpuBS <- block(fgpuB, 2L,4L,2L,4L)
    
    fgpuC <- fgpuAS %*% fgpuBS
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("CPU gpuMatrix Single Precision Matrix Subtraction", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS - BS
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    
    fgpuAS <- block(fgpuA, 2L,4L,2L,4L)
    fgpuBS <- block(fgpuB, 2L,4L,2L,4L)
    fgpuE <- block(fgpuA, 1L,4L,2L,4L)
    
    fgpuC <- fgpuAS - fgpuBS
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fgpuAS - fgpuE)
})

test_that("CPU gpuMatrix Single Precision Scalar Matrix Subtraction", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    
    C <- AS - 1
    C2 <- 1 - AS
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuAS <- block(fgpuA, 2L,4L,2L,4L)
    
    fgpuC <- fgpuAS - 1    
    fgpuC2 <- 1 - fgpuAS
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fgpuC2, "fgpuMatrix")
    expect_equal(fgpuC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Single Precision Unary Scalar Matrix Subtraction", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    C <- -AS
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuAS <- block(fgpuA, 2L,4L,2L,4L)
    
    fgpuC <- -fgpuAS
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Single Precision Matrix Addition", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS + BS
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    
    fgpuAS <- block(fgpuA, 2L,4L,2L,4L)
    fgpuBS <- block(fgpuB, 2L,4L,2L,4L)
    fgpuE <- block(fgpuA, 1L,4L,2L,4L)
    
    fgpuC <- fgpuAS + fgpuBS
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fgpuA + fgpuE)
})

test_that("CPU gpuMatrix Single Precision Scalar Matrix Addition", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    C <- AS + 1
    C2 <- 1 + AS
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuAS <- block(fgpuA, 2L,4L,2L,4L)
    
    fgpuC <- fgpuAS + 1
    fgpuC2 <- 1 + fgpuAS
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fgpuC2, "fgpuMatrix")
    expect_equal(fgpuC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Single Precision Matrix Element-Wise Multiplication", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS * BS
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    
    fgpuAS <- block(fgpuA, 2L,4L,2L,4L)
    fgpuBS <- block(fgpuB, 2L,4L,2L,4L)
    
    fgpuC <- fgpuAS * fgpuBS
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fgpuA * fgpuE)
})

test_that("CPU gpuMatrix Single Precision Scalar Matrix Multiplication", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    C <- AS * 2
    C2 <- 2 * AS
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuAS <- block(fgpuA, 2L,4L,2L,4L)
    
    fgpuC <- fgpuAS * 2
    fgpuC2 <- 2 * fgpuAS
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fgpuC2, "fgpuMatrix")
    expect_equal(fgpuC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Single Precision Matrix Element-Wise Division", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS / BS
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    
    fgpuAS <- block(fgpuA, 2L,4L,2L,4L)
    fgpuBS <- block(fgpuB, 2L,4L,2L,4L)
    fgpuE <- block(fgpuA, 1L,4L,2L,4L)
    
    fgpuC <- fgpuAS / fgpuBS
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fgpuA / fgpuE)
})

test_that("CPU gpuMatrix Single Precision Scalar Matrix Division", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    C <- AS/2
    C2 <- 2/AS
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuAS <- block(fgpuA, 2L,4L,2L,4L)
    
    fgpuC <- fgpuAS/2
    fgpuC2 <- 2/fgpuAS
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fgpuC2, "fgpuMatrix")
    expect_equal(fgpuC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Single Precision Matrix Element-Wise Power", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS ^ BS
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    
    fgpuAS <- block(fgpuA, 2L,4L,2L,4L)
    fgpuBS <- block(fgpuB, 2L,4L,2L,4L)
    fgpuE <- block(fgpuA, 1L,4L,2L,4L)
    
    fgpuC <- fgpuAS ^ fgpuBS
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fgpuA ^ fgpuE)
})

test_that("CPU gpuMatrix Single Precision Scalar Matrix Power", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    C <- AS^2
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuAS <- block(fgpuA, 2L,4L,2L,4L)
    
    fgpuC <- fgpuAS^2
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Single Precision crossprod", {
    
    has_cpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    XS = X[1:2, 2:5]
    YS = Y[1:2, 2:5]
    ZS = Z[2:5, 1:2]
    
    C <- crossprod(XS,YS)
    Cs <- crossprod(XS)
    
    fgpuX <- gpuMatrix(X, type="float")
    fgpuY <- gpuMatrix(Y, type="float")
    fgpuZ <- gpuMatrix(Z, type="float")
    
    fgpuXS <- block(fgpuX, 1L,2L,2L,5L)
    fgpuYS <- block(fgpuY, 1L,2L,2L,5L)
    fgpuZS <- block(fgpuZ, 2L,5L,1L,2L)
    
    fgpuC <- crossprod(fgpuXS, fgpuYS)
    fgpuCs <- crossprod(fgpuXS)
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fgpuCs[,], Cs, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_error(crossprod(fgpuXS, fgpuZS))
})

test_that("CPU gpuMatrix Single Precision tcrossprod", {
    
    has_cpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    XS = X[1:2, 2:5]
    YS = Y[1:2, 2:5]
    ZS = Z[2:5, 1:2]
    
    C <- tcrossprod(XS,YS)
    Cs <- tcrossprod(XS)
    
    fgpuX <- gpuMatrix(X, type="float")
    fgpuY <- gpuMatrix(Y, type="float")
    fgpuZ <- gpuMatrix(Z, type="float")
    
    fgpuXS <- block(fgpuX, 1L,2L,2L,5L)
    fgpuYS <- block(fgpuY, 1L,2L,2L,5L)
    fgpuZS <- block(fgpuZ, 2L,5L,1L,2L)
    
    fgpuC <- tcrossprod(fgpuXS, fgpuYS)
    fgpuCs <- tcrossprod(fgpuXS)
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fgpuCs[,], Cs, tolerance=1e-06, 
                 info="float matrix elements not equivalent") 
    expect_error(crossprod(fgpuXS, fgpuZS))
})

# Double Precision Matrix Block tests

test_that("CPU gpuMatrix Double Precision Block Matrix multiplication", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS %*% BS
    
    dgpuA <- gpuMatrix(A, type="double")
    dgpuB <- gpuMatrix(B, type="double")
    
    dgpuAS <- block(dgpuA, 2L,4L,2L,4L)
    dgpuBS <- block(dgpuB, 2L,4L,2L,4L)
    
    dgpuC <- dgpuAS %*% dgpuBS
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})

test_that("CPU gpuMatrix Double Precision Matrix Subtraction", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS - BS
    
    dgpuA <- gpuMatrix(A, type="double")
    dgpuB <- gpuMatrix(B, type="double")
    
    dgpuAS <- block(dgpuA, 2L,4L,2L,4L)
    dgpuBS <- block(dgpuB, 2L,4L,2L,4L)
    dgpuE <- block(dgpuA, 1L,4L,2L,4L)
    
    dgpuC <- dgpuAS - dgpuBS
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dgpuAS - dgpuE)
})

test_that("CPU gpuMatrix Double Precision Scalar Matrix Subtraction", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    
    C <- AS - 1
    C2 <- 1 - AS
    
    dgpuA <- gpuMatrix(A, type="double")
    dgpuAS <- block(dgpuA, 2L,4L,2L,4L)
    
    dgpuC <- dgpuAS - 1    
    dgpuC2 <- 1 - dgpuAS
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dgpuC2, "dgpuMatrix")
    expect_equal(dgpuC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Double Precision Unary Scalar Matrix Subtraction", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    C <- -AS
    
    dgpuA <- gpuMatrix(A, type="double")
    dgpuAS <- block(dgpuA, 2L,4L,2L,4L)
    
    dgpuC <- -dgpuAS
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Double Precision Matrix Addition", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS + BS
    
    dgpuA <- gpuMatrix(A, type="double")
    dgpuB <- gpuMatrix(B, type="double")
    
    dgpuAS <- block(dgpuA, 2L,4L,2L,4L)
    dgpuBS <- block(dgpuB, 2L,4L,2L,4L)
    dgpuE <- block(dgpuA, 1L,4L,2L,4L)
    
    dgpuC <- dgpuAS + dgpuBS
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dgpuA + dgpuE)
})

test_that("CPU gpuMatrix Double Precision Scalar Matrix Addition", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    C <- AS + 1
    C2 <- 1 + AS
    
    dgpuA <- gpuMatrix(A, type="double")
    dgpuAS <- block(dgpuA, 2L,4L,2L,4L)
    
    dgpuC <- dgpuAS + 1
    dgpuC2 <- 1 + dgpuAS
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dgpuC2, "dgpuMatrix")
    expect_equal(dgpuC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Double Precision Matrix Element-Wise Multiplication", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS * BS
    
    dgpuA <- gpuMatrix(A, type="double")
    dgpuB <- gpuMatrix(B, type="double")
    
    dgpuAS <- block(dgpuA, 2L,4L,2L,4L)
    dgpuBS <- block(dgpuB, 2L,4L,2L,4L)
    
    dgpuC <- dgpuAS * dgpuBS
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dgpuA * dgpuE)
})

test_that("CPU gpuMatrix Double Precision Scalar Matrix Multiplication", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    C <- AS * 2
    C2 <- 2 * AS
    
    dgpuA <- gpuMatrix(A, type="double")
    dgpuAS <- block(dgpuA, 2L,4L,2L,4L)
    
    dgpuC <- dgpuAS * 2
    dgpuC2 <- 2 * dgpuAS
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dgpuC2, "dgpuMatrix")
    expect_equal(dgpuC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Double Precision Matrix Element-Wise Division", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS / BS
    
    dgpuA <- gpuMatrix(A, type="double")
    dgpuB <- gpuMatrix(B, type="double")
    
    dgpuAS <- block(dgpuA, 2L,4L,2L,4L)
    dgpuBS <- block(dgpuB, 2L,4L,2L,4L)
    dgpuE <- block(dgpuA, 1L,4L,2L,4L)
    
    dgpuC <- dgpuAS / dgpuBS
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dgpuA / dgpuE)
})

test_that("CPU gpuMatrix Double Precision Scalar Matrix Division", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    C <- AS/2
    C2 <- 2/AS
    
    dgpuA <- gpuMatrix(A, type="double")
    dgpuAS <- block(dgpuA, 2L,4L,2L,4L)
    
    dgpuC <- dgpuAS/2
    dgpuC2 <- 2/dgpuAS
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dgpuC2, "dgpuMatrix")
    expect_equal(dgpuC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Double Precision Matrix Element-Wise Power", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS ^ BS
    
    dgpuA <- gpuMatrix(A, type="double")
    dgpuB <- gpuMatrix(B, type="double")
    
    dgpuAS <- block(dgpuA, 2L,4L,2L,4L)
    dgpuBS <- block(dgpuB, 2L,4L,2L,4L)
    dgpuE <- block(dgpuA, 1L,4L,2L,4L)
    
    dgpuC <- dgpuAS ^ dgpuBS
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dgpuA ^ dgpuE)
})

test_that("CPU gpuMatrix Double Precision Scalar Matrix Power", {
    
    has_cpu_skip()
    
    AS = A[2:4, 2:4]
    C <- AS^2
    
    dgpuA <- gpuMatrix(A, type="double")
    dgpuAS <- block(dgpuA, 2L,4L,2L,4L)
    
    dgpuC <- dgpuAS^2
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Double Precision crossprod", {
    
    has_cpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    XS = X[1:2, 2:5]
    YS = Y[1:2, 2:5]
    ZS = Z[2:5, 1:2]
    
    C <- crossprod(XS,YS)
    Cs <- crossprod(XS)
    
    dgpuX <- gpuMatrix(X, type="double")
    dgpuY <- gpuMatrix(Y, type="double")
    dgpuZ <- gpuMatrix(Z, type="double")
    
    dgpuXS <- block(dgpuX, 1L,2L,2L,5L)
    dgpuYS <- block(dgpuY, 1L,2L,2L,5L)
    dgpuZS <- block(dgpuZ, 2L,5L,1L,2L)
    
    dgpuC <- crossprod(dgpuXS, dgpuYS)
    dgpuCs <- crossprod(dgpuXS)
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(dgpuCs[,], Cs, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_error(crossprod(dgpuXS, dgpuZS))
})

test_that("CPU gpuMatrix Double Precision tcrossprod", {
    
    has_cpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    XS = X[1:2, 2:5]
    YS = Y[1:2, 2:5]
    ZS = Z[2:5, 1:2]
    
    C <- tcrossprod(XS,YS)
    Cs <- tcrossprod(XS)
    
    dgpuX <- gpuMatrix(X, type="double")
    dgpuY <- gpuMatrix(Y, type="double")
    dgpuZ <- gpuMatrix(Z, type="double")
    
    dgpuXS <- block(dgpuX, 1L,2L,2L,5L)
    dgpuYS <- block(dgpuY, 1L,2L,2L,5L)
    dgpuZS <- block(dgpuZ, 2L,5L,1L,2L)
    
    dgpuC <- tcrossprod(dgpuXS, dgpuYS)
    dgpuCs <- tcrossprod(dgpuXS)
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(dgpuCs[,], Cs, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_error(crossprod(dgpuXS, dgpuZS))
})

# set option back to GPU
options(gpuR.default.device.type = "gpu")
