library(gpuR)
context("gpuMatrix algebra")

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
Aint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
Bint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
B <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
E <- matrix(rnorm(15), nrow=5)



test_that("gpuMatrix Single Precision Matrix multiplication", {
    
    has_gpu_skip()
    
    C <- A %*% B
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    
    fgpuC <- fgpuA %*% fgpuB
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("gpuMatrix Single Precision Matrix Subtraction", {
    
    has_gpu_skip()
    
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

test_that("gpuMatrix Single Precision Matrix Addition", {
    
    has_gpu_skip()
    
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

test_that("gpuMatrix Single Precision Matrix Element-Wise Multiplication", {
    
    has_gpu_skip()
    
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

test_that("gpuMatrix Single Precision Matrix Element-Wise Division", {
    
    has_gpu_skip()
    
    C <- A / B
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    fgpuE <- gpuMatrix(E, type="float")
    
    fgpuC <- fgpuA / fgpuB
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fgpuA * fgpuE)
})

test_that("gpuMatrix Integer Matrix multiplication", {
    
    has_gpu_skip()
    
    Cint <- Aint %*% Bint
    
    igpuA <- gpuMatrix(Aint, type="integer")
    igpuB <- gpuMatrix(Bint, type="integer")
    
    igpuC <- igpuA %*% igpuB
    
    expect_equivalent(igpuC[,], Cint, 
                      info="float matrix elements not equivalent")      
})

test_that("gpuMatrix Integer Matrix Subtraction", {
    
    has_gpu_skip()
    
    Cint <- Aint - Bint
    
    igpuA <- gpuMatrix(Aint, type="integer")
    igpuB <- gpuMatrix(Bint, type="integer")
    
    igpuC <- igpuA - igpuB
    
    expect_is(igpuC, "igpuMatrix")
    expect_equal(igpuC[,], Cint, 
                 info="integer matrix elements not equivalent")  
})

test_that("gpuMatrix Integer Matrix Addition", {
    
    has_gpu_skip()
    
    Cint <- Aint + Bint
    
    igpuA <- gpuMatrix(Aint, type="integer")
    igpuB <- gpuMatrix(Bint, type="integer")
    
    igpuC <- igpuA + igpuB
    
    expect_is(igpuC, "igpuMatrix")
    expect_equal(igpuC[,], Cint,
                 info="integer matrix elements not equivalent")  
})

test_that("gpuMatrix Double Precision Matrix multiplication", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A %*% B
    
    dgpuA <- gpuMatrix(A, type="double")
    dgpuB <- gpuMatrix(B, type="double")
    
    dgpuC <- dgpuA %*% dgpuB
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})

test_that("gpuMatrix Double Precision Matrix Subtraction", {
    
    has_gpu_skip()
    has_double_skip()
    
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

test_that("gpuMatrix Double Precision Matrix Addition", {
    
    has_gpu_skip()
    has_double_skip()
    
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

test_that("gpuMatrix Double Precision Matrix Element-Wise Multiplication", {
    
    has_gpu_skip()
    has_double_skip()
    
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

test_that("gpuMatrix Double Precision Matrix Element-Wise Division", {
    
    has_gpu_skip()
    has_double_skip()
    
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

test_that("gpuMatrix Single Precision crossprod", {
    
    has_gpu_skip()
    
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

test_that("gpuMatrix Double Precision crossprod", {
    
    has_gpu_skip()
    
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

test_that("gpuMatrix Single Precision tcrossprod", {
    
    has_gpu_skip()
    
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

test_that("gpuMatrix Double Precision tcrossprod", {
    
    has_gpu_skip()
    
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
