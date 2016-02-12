library(gpuR)
context("CPU gpuMatrix algebra")

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

# Single Precision tests

test_that("CPU gpuMatrix Single Precision Matrix multiplication", {
    
    has_cpu_skip()
    
    C <- A %*% B
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    
    fgpuC <- fgpuA %*% fgpuB
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("CPU gpuMatrix Single Precision Matrix Subtraction", {
    
    has_cpu_skip()
    
    C <- A - B
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    fgpuE <- gpuMatrix(E, type="float")
    
    fgpuC <- fgpuA - fgpuB
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fgpuA - fgpuE)
})

test_that("CPU gpuMatrix Single Precision Scalar Matrix Subtraction", {
    
    has_cpu_skip()
    
    C <- A - 1
    C2 <- 1 - A
    
    fgpuA <- gpuMatrix(A, type="float")
    
    fgpuC <- fgpuA - 1    
    fgpuC2 <- 1 - fgpuA
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fgpuC2, "fgpuMatrix")
    expect_equal(fgpuC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Single Precision Unary Scalar Matrix Subtraction", {
    
    has_cpu_skip()
    
    C <- -A
    
    fgpuA <- gpuMatrix(A, type="float")
    
    fgpuC <- -fgpuA
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Single Precision Matrix Addition", {
    
    has_cpu_skip()
    
    C <- A + B
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    fgpuE <- gpuMatrix(E, type="float")
    
    fgpuC <- fgpuA + fgpuB
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fgpuA + fgpuE)
})

test_that("CPU gpuMatrix Single Precision Scalar Matrix Addition", {
    
    has_cpu_skip()
    
    C <- A + 1
    C2 <- 1 + A
    
    fgpuA <- gpuMatrix(A, type="float")
    
    fgpuC <- fgpuA + 1
    fgpuC2 <- 1 + fgpuA
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fgpuC2, "fgpuMatrix")
    expect_equal(fgpuC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Single Precision Matrix Element-Wise Multiplication", {
    
    has_cpu_skip()
    
    C <- A * B
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    fgpuE <- gpuMatrix(E, type="float")
    
    fgpuC <- fgpuA * fgpuB
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fgpuA * fgpuE)
})

test_that("CPU gpuMatrix Single Precision Scalar Matrix Multiplication", {
    
    has_cpu_skip()
    
    C <- A * 2
    C2 <- 2 * A
    
    dgpuA <- gpuMatrix(A, type="float")
    
    dgpuC <- dgpuA * 2
    dgpuC2 <- 2 * dgpuA
    
    expect_is(dgpuC, "fgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(dgpuC2, "fgpuMatrix")
    expect_equal(dgpuC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Single Precision Matrix Element-Wise Division", {
    
    has_cpu_skip()
    
    C <- A / B
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    fgpuE <- gpuMatrix(E, type="float")
    
    fgpuC <- fgpuA / fgpuB
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fgpuA / fgpuE)
})

test_that("CPU gpuMatrix Single Precision Scalar Matrix Division", {
    
    has_cpu_skip()
    
    C <- A/2
    C2 <- 2/A
    
    dgpuA <- gpuMatrix(A, type="float")
    
    dgpuC <- dgpuA/2
    dgpuC2 <- 2/dgpuA
    
    expect_is(dgpuC, "fgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(dgpuC2, "fgpuMatrix")
    expect_equal(dgpuC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Single Precision Matrix Element-Wise Power", {
    
    has_cpu_skip()
    
    C <- A ^ B
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    fgpuE <- gpuMatrix(E, type="float")
    
    fgpuC <- fgpuA ^ fgpuB
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fgpuA ^ fgpuE)
})

test_that("CPU gpuMatrix Single Precision Scalar Matrix Power", {
    
    has_cpu_skip()
    
    C <- A^2
    
    dgpuA <- gpuMatrix(A, type="float")
    
    dgpuC <- dgpuA^2
    
    expect_is(dgpuC, "fgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Single Precision crossprod", {
    
    has_cpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    
    C <- crossprod(X,Y)
    Cs <- crossprod(X)
    
    fgpuX <- gpuMatrix(X, type="float")
    fgpuY <- gpuMatrix(Y, type="float")
    fgpuZ <- gpuMatrix(Z, type="float")
    
    fgpuC <- crossprod(fgpuX, fgpuY)
    fgpuCs <- crossprod(fgpuX)
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fgpuCs[,], Cs, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_error(crossprod(fgpuX, fgpuZ))
})

test_that("CPU gpuMatrix Single Precision tcrossprod", {
    
    has_cpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    
    C <- tcrossprod(X,Y)
    Cs <- tcrossprod(X)
    
    fgpuX <- gpuMatrix(X, type="float")
    fgpuY <- gpuMatrix(Y, type="float")
    fgpuZ <- gpuMatrix(Z, type="float")
    
    fgpuC <- tcrossprod(fgpuX, fgpuY)
    fgpuCs <- tcrossprod(fgpuX)
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fgpuCs[,], Cs, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_error(crossprod(fgpuX, fgpuZ))
})

# Integer tests

test_that("CPU gpuMatrix Integer Matrix multiplication", {
    
    has_cpu_skip()
    
    Cint <- Aint %*% Bint
    
    igpuA <- gpuMatrix(Aint, type="integer")
    igpuB <- gpuMatrix(Bint, type="integer")
    
    igpuC <- igpuA %*% igpuB
    
    expect_equivalent(igpuC[,], Cint, 
                      info="integer matrix elements not equivalent")      
})

test_that("CPU gpuMatrix Integer Matrix Subtraction", {
    
    has_cpu_skip()
    
    Cint <- Aint - Bint
    
    igpuA <- gpuMatrix(Aint, type="integer")
    igpuB <- gpuMatrix(Bint, type="integer")
    
    igpuC <- igpuA - igpuB
    
    expect_is(igpuC, "igpuMatrix")
    expect_equal(igpuC[,], Cint, 
                 info="integer matrix elements not equivalent")  
})

test_that("CPU gpuMatrix Integer Matrix Addition", {
    
    has_cpu_skip()
    
    Cint <- Aint + Bint
    
    igpuA <- gpuMatrix(Aint, type="integer")
    igpuB <- gpuMatrix(Bint, type="integer")
    
    igpuC <- igpuA + igpuB
    
    expect_is(igpuC, "igpuMatrix")
    expect_equal(igpuC[,], Cint,
                 info="integer matrix elements not equivalent")  
})

# Double Precision tests

test_that("CPU gpuMatrix Double Precision Matrix multiplication", {
    
    has_cpu_skip()
    
    
    C <- A %*% B
    
    dgpuA <- gpuMatrix(A, type="double")
    dgpuB <- gpuMatrix(B, type="double")
    
    dgpuC <- dgpuA %*% dgpuB
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})

test_that("CPU gpuMatrix Double Precision Matrix Subtraction", {
    
    has_cpu_skip()
    
    
    C <- A - B
    
    dgpuA <- gpuMatrix(A, type="double")
    dgpuB <- gpuMatrix(B, type="double")
    dgpuE <- gpuMatrix(E, type="double")
    
    dgpuC <- dgpuA - dgpuB
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dgpuA - dgpuE)
})

test_that("CPU gpuMatrix Double Precision Matrix Addition", {
    
    has_cpu_skip()
    
    
    C <- A + B
    
    dgpuA <- gpuMatrix(A, type="double")
    dgpuB <- gpuMatrix(B, type="double")
    dgpuE <- gpuMatrix(E, type="double")
    
    dgpuC <- dgpuA + dgpuB
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dgpuA + dgpuE)
})

test_that("CPU gpuMatrix Double Precision Scalar Matrix Addition", {
    
    has_cpu_skip()
    
    
    C <- A + 1
    C2 <- 1 + A
    
    dgpuA <- gpuMatrix(A, type="double")
    
    dgpuC <- dgpuA + 1
    dgpuC2 <- 1 + dgpuA
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dgpuC2, "dgpuMatrix")
    expect_equal(dgpuC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Double Precision Scalar Matrix Subtraction", {
    
    has_cpu_skip()
    
    
    C <- A - 1
    C2 <- 1 - A
    
    dgpuA <- gpuMatrix(A, type="double")
    
    dgpuC <- dgpuA - 1
    dgpuC2 <- 1 - dgpuA
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dgpuC2, "dgpuMatrix")
    expect_equal(dgpuC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Double Precision Unary Matrix Subtraction", {
    
    has_cpu_skip()
    
    
    C <- -A
    
    fgpuA <- gpuMatrix(A, type="double")
    
    fgpuC <- -fgpuA
    
    expect_is(fgpuC, "dgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Double Precision Matrix Element-Wise Multiplication", {
    
    has_cpu_skip()
    
    
    C <- A * B
    
    dgpuA <- gpuMatrix(A, type="double")
    dgpuB <- gpuMatrix(B, type="double")
    dgpuE <- gpuMatrix(E, type="double")
    
    dgpuC <- dgpuA * dgpuB
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dgpuA * dgpuE)
})

test_that("CPU gpuMatrix Double Precision Scalar Matrix Multiplication", {
    
    has_cpu_skip()
    
    
    C <- A * 2
    C2 <- 2 * A
    
    dgpuA <- gpuMatrix(A, type="double")
    
    dgpuC <- dgpuA * 2
    dgpuC2 <- 2 * dgpuA
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dgpuC2, "dgpuMatrix")
    expect_equal(dgpuC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Double Precision Matrix Element-Wise Division", {
    
    has_cpu_skip()
    
    
    C <- A / B
    
    dgpuA <- gpuMatrix(A, type="double")
    dgpuB <- gpuMatrix(B, type="double")
    dgpuE <- gpuMatrix(E, type="double")
    
    dgpuC <- dgpuA / dgpuB
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dgpuA * dgpuE)
})

test_that("CPU gpuMatrix Double Precision Scalar Matrix Division", {
    
    has_cpu_skip()
    
    
    C <- A/2
    C2 <- 2/A
    
    dgpuA <- gpuMatrix(A, type="double")
    
    dgpuC <- dgpuA/2
    dgpuC2 <- 2/dgpuA
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dgpuC2, "dgpuMatrix")
    expect_equal(dgpuC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Double Precision Matrix Element-Wise Power", {
    
    has_cpu_skip()
    
    
    C <- A ^ B
    
    fgpuA <- gpuMatrix(A, type="double")
    fgpuB <- gpuMatrix(B, type="double")
    fgpuE <- gpuMatrix(E, type="double")
    
    fgpuC <- fgpuA ^ fgpuB
    
    expect_is(fgpuC, "dgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(fgpuA ^ fgpuE)
})

test_that("CPU gpuMatrix Double Precision Scalar Matrix Power", {
    
    has_cpu_skip()
    
    
    C <- A^2
    
    dgpuA <- gpuMatrix(A, type="double")
    
    dgpuC <- dgpuA^2
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("CPU gpuMatrix Double Precision crossprod", {
    
    has_cpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    
    C <- crossprod(X,Y)
    Cs <- crossprod(X)
    
    fgpuX <- gpuMatrix(X, type="double")
    fgpuY <- gpuMatrix(Y, type="double")
    fgpuZ <- gpuMatrix(Z, type="double")
    
    fgpuC <- crossprod(fgpuX, fgpuY)
    fgpuCs <- crossprod(fgpuX)
    
    expect_is(fgpuC, "dgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fgpuCs[,], Cs, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_error(crossprod(fgpuX, fgpuZ))
})

test_that("CPU gpuMatrix Double Precision tcrossprod", {
    
    has_cpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    
    C <- tcrossprod(X,Y)
    Cs <- tcrossprod(X)
    
    fgpuX <- gpuMatrix(X, type="double")
    fgpuY <- gpuMatrix(Y, type="double")
    fgpuZ <- gpuMatrix(Z, type="double")
    
    fgpuC <- tcrossprod(fgpuX, fgpuY)
    fgpuCs <- tcrossprod(fgpuX)
    
    expect_is(fgpuC, "dgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fgpuCs[,], Cs, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_error(crossprod(fgpuX, fgpuZ))
})



# set option back to GPU
options(gpuR.default.device.type = "gpu")
