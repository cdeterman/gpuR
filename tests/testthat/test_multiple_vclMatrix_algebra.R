library(gpuR)
context("Multiple GPU vclMatrix algebra")

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
Aint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
Bint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
B <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
E <- matrix(rnorm(15), nrow=5)

# Single Precision Tests

test_that("vclMatrix Single Precision Matrix Multiplication", {
    
    has_multiple_gpu_skip()
    
    C <- A %*% B
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="float")
    fvclB2 <- vclMatrix(B, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA %*% fvclB
    fvclC2 <- fvclA2 %*% fvclB2
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")
    expect_equal(fvclC2@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
})

test_that("vclMatrix Single Precision Matrix Subtraction", {
    
    has_multiple_gpu_skip()
    
    C <- A - B
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="float")
    fvclB2 <- vclMatrix(B, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA - fvclB
    fvclC2 <- fvclA2 - fvclB2
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fvclC2@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
})

test_that("vclMatrix Single Precision Scalar Matrix Subtraction", {
    
    has_multiple_gpu_skip()
    
    C <- A - 1
    C2 <- 1 - A
    
    fvclA <- vclMatrix(A, type="float")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA - 1    
    fvclC2 <- 1 - fvclA
    
    fvclC3 <- fvclA2 - 1    
    fvclC4 <- 1 - fvclA2
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    
    expect_is(fvclC3, "fvclMatrix")
    expect_equal(fvclC3[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC3@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_is(fvclC4, "fvclMatrix")
    expect_equal(fvclC4[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC4@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
})

test_that("vclMatrix Single Precision Unary Scalar Matrix Subtraction", {
    
    has_multiple_gpu_skip()
    
    C <- -A
    
    fvclA <- vclMatrix(A, type="float")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- -fvclA
    fvclC2 <- -fvclA2
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC2@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
})

test_that("vclMatrix Single Precision Matrix Addition", {
    
    has_multiple_gpu_skip()
    
    C <- A + B
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="float")
    fvclB2 <- vclMatrix(B, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA + fvclB
    fvclC2 <- fvclA2 + fvclB2
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fvclC2@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
})

test_that("vclMatrix Single Precision Scalar Matrix Addition", {
    
    has_multiple_gpu_skip()
    
    C <- A + 1
    C2 <- 1 + A
    
    fvclA <- vclMatrix(A, type="float")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA + 1
    fvclC2 <- 1 + fvclA
    fvclC3 <- fvclA2 + 1
    fvclC4 <- 1 + fvclA2
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fvclC3, "fvclMatrix")
    expect_equal(fvclC3[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fvclC4, "fvclMatrix")
    expect_equal(fvclC4[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC3@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(fvclC4@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
})

test_that("vclMatrix Single Precision Matrix Element-Wise Multiplication", {
    
    has_multiple_gpu_skip()
    
    C <- A * B
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    fvclE <- vclMatrix(E, type="float")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="float")
    fvclB2 <- vclMatrix(B, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA * fvclB
    fvclC2 <- fvclA2 * fvclB2
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fvclA * fvclE,
                 info="should return error with different dimensions")
    expect_equal(fvclC2@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
})

test_that("vclMatrix Single Precision Scalar Matrix Multiplication", {
    
    has_multiple_gpu_skip()
    
    C <- A * 2
    C2 <- 2 * A
    
    fvclA <- vclMatrix(A, type="float")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA * 2
    fvclC2 <- 2 * fvclA
    fvclC3 <- fvclA2 * 2
    fvclC4 <- 2 * fvclA2
    
    expect_is(fvclC, "fvclMatrix")
    expect_is(fvclC2, "fvclMatrix")
    expect_is(fvclC3, "fvclMatrix") 
    expect_is(fvclC4, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC3[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")
    expect_equal(fvclC4[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC3@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(fvclC4@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
})

test_that("vclMatrix Single Precision Matrix Element-Wise Division", {
    
    has_multiple_gpu_skip()
    
    C <- A / B
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    fvclE <- vclMatrix(E, type="float")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="float")
    fvclB2 <- vclMatrix(B, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA / fvclB
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fvclA * fvclE,
                 info="should return error with different dimensions")
    expect_equal(fvclC2@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
})

test_that("vclMatrix Single Precision Scalar Matrix Division", {
    
    has_multiple_gpu_skip()
    
    C <- A/2
    C2 <- 2/A
    
    fvclA <- vclMatrix(A, type="float")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA / 2
    fvclC2 <- 2 / fvclA
    fvclC3 <- fvclA2 / 2
    fvclC4 <- 2 / fvclA2
    
    expect_is(fvclC, "fvclMatrix")
    expect_is(fvclC2, "fvclMatrix")
    expect_is(fvclC3, "fvclMatrix")
    expect_is(fvclC4, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC3[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC4[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC3@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(fvclC4@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
})

test_that("vclMatrix Single Precision Matrix Element-Wise Power", {
    
    has_multiple_gpu_skip()
    
    C <- A ^ B
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    fvclE <- vclMatrix(E, type="float")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="float")
    fvclB2 <- vclMatrix(B, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA ^ fvclB
    fvclC2 <- fvclA2 ^ fvclB2
    
    expect_is(fvclC, "fvclMatrix")
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fvclC2[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fvclA ^ fvclE)
})

test_that("vclMatrix Single Precision Scalar Matrix Power", {
    
    has_multiple_gpu_skip()
    
    C <- A^2
    
    fvclA <- vclMatrix(A, type="float")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA^2
    fvclC2 <- fvclA2^2
    
    expect_is(fvclC, "fvclMatrix")
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC2[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("vclMatrix Single Precision crossprod", {
    
    has_multiple_gpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    
    C <- crossprod(X,Y)
    Cs <- crossprod(X)
    
    fvclX <- vclMatrix(X, type="float")
    fvclY <- vclMatrix(Y, type="float")
    fvclZ <- vclMatrix(Z, type="float")
    
    setContext(2L)
    
    fvclX2 <- vclMatrix(X, type="float")
    fvclY2 <- vclMatrix(Y, type="float")
    fvclZ2 <- vclMatrix(Z, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- crossprod(fvclX, fvclY)
    fvclCs <- crossprod(fvclX)
    fvclC2 <- crossprod(fvclX2, fvclY2)
    fvclCs2 <- crossprod(fvclX2)
    
    expect_is(fvclC, "fvclMatrix")
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fvclCs[,], Cs, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC2[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fvclCs2[,], Cs, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_error(crossprod(fvclX, fvclZ))
})

test_that("vclMatrix Single Precision tcrossprod", {
    
    has_multiple_gpu_skip()
    
    X <- matrix(rnorm(10), nrow=5)
    Y <- matrix(rnorm(10), nrow=5)
    Z <- matrix(rnorm(15), nrow=5)
    
    C <- tcrossprod(X,Y)
    Cs <- tcrossprod(X)
    
    fvclX <- vclMatrix(X, type="float")
    fvclY <- vclMatrix(Y, type="float")
    fvclZ <- vclMatrix(Z, type="float")
    
    setContext(2L)
    
    fvclX2 <- vclMatrix(X, type="float")
    fvclY2 <- vclMatrix(Y, type="float")
    fvclZ2 <- vclMatrix(Z, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- tcrossprod(fvclX, fvclY)
    fvclCs <- tcrossprod(fvclX)
    fvclC2 <- tcrossprod(fvclX2, fvclY2)
    fvclCs2 <- tcrossprod(fvclX2)
    
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fvclCs[,], Cs, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC2[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fvclCs2[,], Cs, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_error(tcrossprod(fvclX, fvclZ))
})

test_that("vclMatrix Single Precision transpose", {
    
    has_multiple_gpu_skip()
    
    At <- t(A)
    
    fgpuA <- vclMatrix(A, type="float")
    
    setContext(2L)
    
    fgpuA2 <- vclMatrix(A, type="float")
    
    fgpuAt <- t(fgpuA)
    fgpuAt2 <- t(fgpuA2)
    
    expect_is(fgpuAt, "fvclMatrix")
    expect_is(fgpuAt2, "fvclMatrix")
    expect_equal(fgpuAt[,], At, tolerance=1e-07, 
                 info="transposed float matrix elements not equivalent") 
    expect_equal(fgpuAt2[,], At, tolerance=1e-07, 
                 info="transposed float matrix elements not equivalent") 
})

# Double Precision Tests

test_that("vclMatrix Double Precision Matrix Multiplication", {
    
    has_multiple_double_skip()
    
    C <- A %*% B
    
    fvclA <- vclMatrix(A, type="double")
    fvclB <- vclMatrix(B, type="double")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="double")
    fvclB2 <- vclMatrix(B, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA %*% fvclB
    fvclC2 <- fvclA2 %*% fvclB2
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")
    expect_equal(fvclC2@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
})

test_that("vclMatrix Double Precision Matrix Subtraction", {
    
    has_multiple_double_skip()
    
    C <- A - B
    
    fvclA <- vclMatrix(A, type="double")
    fvclB <- vclMatrix(B, type="double")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="double")
    fvclB2 <- vclMatrix(B, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA - fvclB
    fvclC2 <- fvclA2 - fvclB2
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fvclC2@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
})

test_that("vclMatrix Double Precision Scalar Matrix Subtraction", {
    
    has_multiple_double_skip()
    
    C <- A - 1
    C2 <- 1 - A
    
    fvclA <- vclMatrix(A, type="double")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA - 1    
    fvclC2 <- 1 - fvclA
    
    fvclC3 <- fvclA2 - 1    
    fvclC4 <- 1 - fvclA2
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    
    expect_is(fvclC3, "fvclMatrix")
    expect_equal(fvclC3[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC3@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_is(fvclC4, "fvclMatrix")
    expect_equal(fvclC4[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC4@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
})

test_that("vclMatrix Double Precision Unary Scalar Matrix Subtraction", {
    
    has_multiple_double_skip()
    
    C <- -A
    
    fvclA <- vclMatrix(A, type="double")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- -fvclA
    fvclC2 <- -fvclA2
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC2@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
})

test_that("vclMatrix Double Precision Matrix Addition", {
    
    has_multiple_double_skip()
    
    C <- A + B
    
    fvclA <- vclMatrix(A, type="double")
    fvclB <- vclMatrix(B, type="double")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="double")
    fvclB2 <- vclMatrix(B, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA + fvclB
    fvclC2 <- fvclA2 + fvclB2
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fvclC2@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
})

test_that("vclMatrix Double Precision Scalar Matrix Addition", {
    
    has_multiple_double_skip()
    
    C <- A + 1
    C2 <- 1 + A
    
    fvclA <- vclMatrix(A, type="double")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA + 1
    fvclC2 <- 1 + fvclA
    fvclC3 <- fvclA2 + 1
    fvclC4 <- 1 + fvclA2
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(fvclC3, "fvclMatrix")
    expect_equal(fvclC3[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(fvclC4, "fvclMatrix")
    expect_equal(fvclC4[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC3@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(fvclC4@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
})

test_that("vclMatrix Double Precision Matrix Element-Wise Multiplication", {
    
    has_multiple_double_skip()
    
    C <- A * B
    
    fvclA <- vclMatrix(A, type="double")
    fvclB <- vclMatrix(B, type="double")
    fvclE <- vclMatrix(E, type="double")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="double")
    fvclB2 <- vclMatrix(B, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA * fvclB
    fvclC2 <- fvclA2 * fvclB2
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(fvclA * fvclE,
                 info="should return error with different dimensions")
    expect_equal(fvclC2@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
})

test_that("vclMatrix Double Precision Scalar Matrix Multiplication", {
    
    has_multiple_double_skip()
    
    C <- A * 2
    C2 <- 2 * A
    
    fvclA <- vclMatrix(A, type="double")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA * 2
    fvclC2 <- 2 * fvclA
    fvclC3 <- fvclA2 * 2
    fvclC4 <- 2 * fvclA2
    
    expect_is(fvclC, "fvclMatrix")
    expect_is(fvclC2, "fvclMatrix")
    expect_is(fvclC3, "fvclMatrix") 
    expect_is(fvclC4, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC3[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")
    expect_equal(fvclC4[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC3@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(fvclC4@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
})

test_that("vclMatrix Double Precision Matrix Element-Wise Division", {
    
    has_multiple_double_skip()
    
    C <- A / B
    
    fvclA <- vclMatrix(A, type="double")
    fvclB <- vclMatrix(B, type="double")
    fvclE <- vclMatrix(E, type="double")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="double")
    fvclB2 <- vclMatrix(B, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA / fvclB
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(fvclA * fvclE,
                 info="should return error with different dimensions")
    expect_equal(fvclC2@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
})

test_that("vclMatrix Double Precision Scalar Matrix Division", {
    
    has_multiple_double_skip()
    
    C <- A/2
    C2 <- 2/A
    
    fvclA <- vclMatrix(A, type="double")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA / 2
    fvclC2 <- 2 / fvclA
    fvclC3 <- fvclA2 / 2
    fvclC4 <- 2 / fvclA2
    
    expect_is(fvclC, "fvclMatrix")
    expect_is(fvclC2, "fvclMatrix")
    expect_is(fvclC3, "fvclMatrix")
    expect_is(fvclC4, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC3[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC4[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC3@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(fvclC4@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
})

test_that("vclMatrix Double Precision Matrix Element-Wise Power", {
    
    has_multiple_double_skip()
    
    C <- A ^ B
    
    fvclA <- vclMatrix(A, type="double")
    fvclB <- vclMatrix(B, type="double")
    fvclE <- vclMatrix(E, type="double")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="double")
    fvclB2 <- vclMatrix(B, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA ^ fvclB
    fvclC2 <- fvclA2 ^ fvclB2
    
    expect_is(fvclC, "fvclMatrix")
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fvclC2[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(fvclA ^ fvclE)
})

test_that("vclMatrix Double Precision Scalar Matrix Power", {
    
    has_multiple_double_skip()
    
    C <- A^2
    
    fvclA <- vclMatrix(A, type="double")
    
    setContext(2L)
    
    fvclA2 <- vclMatrix(A, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA^2
    fvclC2 <- fvclA2^2
    
    expect_is(fvclC, "fvclMatrix")
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC2[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("vclMatrix Double Precision crossprod", {
    
    has_multiple_double_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    
    C <- crossprod(X,Y)
    Cs <- crossprod(X)
    
    fvclX <- vclMatrix(X, type="double")
    fvclY <- vclMatrix(Y, type="double")
    fvclZ <- vclMatrix(Z, type="double")
    
    setContext(2L)
    
    fvclX2 <- vclMatrix(X, type="double")
    fvclY2 <- vclMatrix(Y, type="double")
    fvclZ2 <- vclMatrix(Z, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- crossprod(fvclX, fvclY)
    fvclCs <- crossprod(fvclX)
    fvclC2 <- crossprod(fvclX2, fvclY2)
    fvclCs2 <- crossprod(fvclX2)
    
    expect_is(fvclC, "fvclMatrix")
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fvclCs[,], Cs, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC2[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fvclCs2[,], Cs, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_error(crossprod(fvclX, fvclZ))
})

test_that("vclMatrix Double Precision tcrossprod", {
    
    has_multiple_double_skip()
    
    X <- matrix(rnorm(10), nrow=5)
    Y <- matrix(rnorm(10), nrow=5)
    Z <- matrix(rnorm(15), nrow=5)
    
    C <- tcrossprod(X,Y)
    Cs <- tcrossprod(X)
    
    fvclX <- vclMatrix(X, type="double")
    fvclY <- vclMatrix(Y, type="double")
    fvclZ <- vclMatrix(Z, type="double")
    
    setContext(2L)
    
    fvclX2 <- vclMatrix(X, type="double")
    fvclY2 <- vclMatrix(Y, type="double")
    fvclZ2 <- vclMatrix(Z, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    fvclC <- tcrossprod(fvclX, fvclY)
    fvclCs <- tcrossprod(fvclX)
    fvclC2 <- tcrossprod(fvclX2, fvclY2)
    fvclCs2 <- tcrossprod(fvclX2)
    
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fvclCs[,], Cs, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC2[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fvclCs2[,], Cs, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_error(tcrossprod(fvclX, fvclZ))
})

test_that("vclMatrix Double Precision transpose", {
    
    has_multiple_double_skip()
    
    At <- t(A)
    
    fgpuA <- vclMatrix(A, type="double")
    
    setContext(2L)
    
    fgpuA2 <- vclMatrix(A, type="double")
    
    fgpuAt <- t(fgpuA)
    fgpuAt2 <- t(fgpuA2)
    
    expect_is(fgpuAt, "fvclMatrix")
    expect_is(fgpuAt2, "fvclMatrix")
    expect_equal(fgpuAt[,], At, tolerance=.Machine$double.eps ^ 0.5, 
                 info="transposed double matrix elements not equivalent") 
    expect_equal(fgpuAt2[,], At, tolerance=.Machine$double.eps ^ 0.5, 
                 info="transposed double matrix elements not equivalent") 
})
