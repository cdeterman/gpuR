library(gpuR)
context("Switching GPUs vclMatrix algebra")

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

test_that("Switching GPUs vclMatrix Single Precision Matrix Multiplication", {
    
    has_multiple_gpu_skip()
    
    C <- A %*% B
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- fvclA %*% fvclB

    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Single Precision Matrix Subtraction", {
    
    has_multiple_gpu_skip()
    
    C <- A - B
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- fvclA - fvclB
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Single Precision Scalar Matrix Subtraction", {
    
    has_multiple_gpu_skip()
    
    C <- A - 1
    C2 <- 1 - A
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- fvclA - 1    
    fvclC2 <- 1 - fvclA
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(fvclC2@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Single Precision Unary Scalar Matrix Subtraction", {
    
    has_multiple_gpu_skip()
    
    C <- -A
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- -fvclA
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Single Precision Matrix Addition", {
    
    has_multiple_gpu_skip()
    
    C <- A + B
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- fvclA + fvclB
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Single Precision Scalar Matrix Addition", {
    
    has_multiple_gpu_skip()
    
    C <- A + 1
    C2 <- 1 + A
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- fvclA + 1
    fvclC2 <- 1 + fvclA
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(fvclC2@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Single Precision Matrix Element-Wise Multiplication", {
    
    has_multiple_gpu_skip()
    
    C <- A * B
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- fvclA * fvclB
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Single Precision Scalar Matrix Multiplication", {
    
    has_multiple_gpu_skip()
    
    C <- A * 2
    C2 <- 2 * A
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- fvclA * 2
    fvclC2 <- 2 * fvclA
    
    expect_is(fvclC, "fvclMatrix")
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(fvclC2@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Single Precision Matrix Element-Wise Division", {
    
    has_multiple_gpu_skip()
    
    C <- A / B
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- fvclA / fvclB
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Single Precision Scalar Matrix Division", {
    
    has_multiple_gpu_skip()
    
    C <- A/2
    C2 <- 2/A
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- fvclA / 2
    fvclC2 <- 2 / fvclA
    
    expect_is(fvclC, "fvclMatrix")
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(fvclC2@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Single Precision Matrix Element-Wise Power", {
    
    has_multiple_gpu_skip()
    
    C <- A ^ B
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- fvclA ^ fvclB
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Single Precision Scalar Matrix Power", {
    
    has_multiple_gpu_skip()
    
    C <- A^2
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- fvclA^2
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Single Precision crossprod", {
    
    has_multiple_gpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    
    C <- crossprod(X,Y)
    Cs <- crossprod(X)
    
    setContext(2L)
    
    fvclX <- vclMatrix(X, type="float")
    fvclY <- vclMatrix(Y, type="float")
    fvclZ <- vclMatrix(Z, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- crossprod(fvclX, fvclY)
    fvclCs <- crossprod(fvclX)
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fvclCs[,], Cs, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_error(crossprod(fvclX, fvclZ))
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(fvclCs@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Single Precision tcrossprod", {
    
    has_multiple_gpu_skip()
    
    X <- matrix(rnorm(10), nrow=5)
    Y <- matrix(rnorm(10), nrow=5)
    Z <- matrix(rnorm(15), nrow=5)
    
    C <- tcrossprod(X,Y)
    Cs <- tcrossprod(X)
    
    setContext(2L)
    
    fvclX <- vclMatrix(X, type="float")
    fvclY <- vclMatrix(Y, type="float")
    fvclZ <- vclMatrix(Z, type="float")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- tcrossprod(fvclX, fvclY)
    fvclCs <- tcrossprod(fvclX)
    
    expect_is(fvclC, "fvclMatrix")
    expect_is(fvclCs, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fvclCs[,], Cs, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_error(tcrossprod(fvclX, fvclZ))
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(fvclCs@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Single Precision transpose", {
    
    has_multiple_gpu_skip()
    
    At <- t(A)
    
    setContext(2L)
    
    fgpuA <- vclMatrix(A, type="float")
    
    setContext(1L)
    
    fgpuAt <- t(fgpuA)
    
    expect_is(fgpuAt, "fvclMatrix")
    expect_equal(fgpuAt[,], At, tolerance=1e-07, 
                 info="transposed float matrix elements not equivalent") 
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

# Double Precision Tests

test_that("Switching GPUs vclMatrix Double Precision Matrix Multiplication", {
    
    has_multiple_double_skip()
    
    C <- A %*% B
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="double")
    fvclB <- vclMatrix(B, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- fvclA %*% fvclB
    
    expect_is(fvclC, "dvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Double Precision Matrix Subtraction", {
    
    has_multiple_double_skip()
    
    C <- A - B
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="double")
    fvclB <- vclMatrix(B, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- fvclA - fvclB
    
    expect_is(fvclC, "dvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Double Precision Scalar Matrix Subtraction", {
    
    has_multiple_double_skip()
    
    C <- A - 1
    C2 <- 1 - A
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- fvclA - 1    
    fvclC2 <- 1 - fvclA
    
    expect_is(fvclC, "dvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(fvclC2, "dvclMatrix")
    expect_equal(fvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(fvclC2@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Double Precision Unary Scalar Matrix Subtraction", {
    
    has_multiple_double_skip()
    
    C <- -A
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- -fvclA
    
    expect_is(fvclC, "dvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Double Precision Matrix Addition", {
    
    has_multiple_double_skip()
    
    C <- A + B
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="double")
    fvclB <- vclMatrix(B, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- fvclA + fvclB
    
    expect_is(fvclC, "dvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Double Precision Scalar Matrix Addition", {
    
    has_multiple_double_skip()
    
    C <- A + 1
    C2 <- 1 + A
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- fvclA + 1
    fvclC2 <- 1 + fvclA
    
    expect_is(fvclC, "dvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(fvclC2, "dvclMatrix")
    expect_equal(fvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(fvclC2@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Double Precision Matrix Element-Wise Multiplication", {
    
    has_multiple_double_skip()
    
    C <- A * B
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="double")
    fvclB <- vclMatrix(B, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- fvclA * fvclB
    
    expect_is(fvclC, "dvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Double Precision Scalar Matrix Multiplication", {
    
    has_multiple_double_skip()
    
    C <- A * 2
    C2 <- 2 * A
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- fvclA * 2
    fvclC2 <- 2 * fvclA
    
    expect_is(fvclC, "dvclMatrix")
    expect_is(fvclC2, "dvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(fvclC2@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Double Precision Matrix Element-Wise Division", {
    
    has_multiple_double_skip()
    
    C <- A / B
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="double")
    fvclB <- vclMatrix(B, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- fvclA / fvclB
    
    expect_is(fvclC, "dvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Double Precision Scalar Matrix Division", {
    
    has_multiple_double_skip()
    
    C <- A/2
    C2 <- 2/A
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- fvclA / 2
    fvclC2 <- 2 / fvclA
    
    expect_is(fvclC, "dvclMatrix")
    expect_is(fvclC2, "dvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(fvclC2@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Double Precision Matrix Element-Wise Power", {
    
    has_multiple_double_skip()
    
    C <- A ^ B
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="double")
    fvclB <- vclMatrix(B, type="double")
    
    setContext(1L)
    
    expect_equal(currentContext(), 1L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA ^ fvclB
    
    expect_is(fvclC, "dvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Double Precision Scalar Matrix Power", {
    
    has_multiple_double_skip()
    
    C <- A^2
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="double")
    
    setContext(1L)
    
    expect_equal(currentContext(), 1L, 
                 info = "context hasn't been changed")
    
    fvclC <- fvclA^2
    
    expect_is(fvclC, "dvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Double Precision crossprod", {
    
    has_multiple_double_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    
    C <- crossprod(X,Y)
    Cs <- crossprod(X)
    
    setContext(2L)
    
    fvclX <- vclMatrix(X, type="double")
    fvclY <- vclMatrix(Y, type="double")
    fvclZ <- vclMatrix(Z, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- crossprod(fvclX, fvclY)
    fvclCs <- crossprod(fvclX)
    
    expect_is(fvclC, "dvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fvclCs[,], Cs, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_error(crossprod(fvclX, fvclZ))
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(fvclCs@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Double Precision tcrossprod", {
    
    has_multiple_double_skip()
    
    X <- matrix(rnorm(10), nrow=5)
    Y <- matrix(rnorm(10), nrow=5)
    Z <- matrix(rnorm(15), nrow=5)
    
    C <- tcrossprod(X,Y)
    Cs <- tcrossprod(X)
    
    setContext(2L)
    
    fvclX <- vclMatrix(X, type="double")
    fvclY <- vclMatrix(Y, type="double")
    fvclZ <- vclMatrix(Z, type="double")
    
    expect_equal(currentContext(), 2L, 
                 info = "context hasn't been changed")
    
    setContext(1L)
    
    fvclC <- tcrossprod(fvclX, fvclY)
    fvclCs <- tcrossprod(fvclX)
    
    expect_is(fvclC, "dvclMatrix")
    expect_is(fvclCs, "dvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fvclCs[,], Cs, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_error(tcrossprod(fvclX, fvclZ))
    expect_equal(fvclC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(fvclCs@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Double Precision transpose", {
    
    has_multiple_double_skip()
    
    At <- t(A)
    
    setContext(2L)
    
    fgpuA <- vclMatrix(A, type="double")
    
    setContext(1L)
    
    fgpuAt <- t(fgpuA)
    
    expect_is(fgpuAt, "dvclMatrix")
    expect_equal(fgpuAt[,], At, tolerance=.Machine$double.eps ^ 0.5, 
                 info="transposed double matrix elements not equivalent") 
    expect_equal(fgpuAt@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})
