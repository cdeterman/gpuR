library(gpuR)
context("Switching GPU vclMatrixBlock algebra")

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

test_that("Switching GPU vclMatrixBlock Single Precision Block Matrix multiplication", {
    
    has_multiple_gpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS %*% BS
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    setContext(1L)
    
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    fvclBS <- block(fvclB, 2L,4L,2L,4L)
    
    fvclC <- fvclAS %*% fvclBS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Single Precision Matrix Subtraction", {
    
    has_multiple_gpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS - BS
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    setContext(1L)
    
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    fvclBS <- block(fvclB, 2L,4L,2L,4L)
    fvclE <- block(fvclA, 1L,4L,2L,4L)
    
    fvclC <- fvclAS - fvclBS

    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fvclAS - fvclE)
    expect_equal(fvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Single Precision Scalar Matrix Subtraction", {
    
    has_multiple_gpu_skip()
    
    AS = A[2:4, 2:4]
    
    C <- AS - 1
    C2 <- 1 - AS
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    
    setContext(1L)
    
    fvclC <- fvclAS - 1    
    fvclC2 <- 1 - fvclAS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(fvclC2@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Single Precision Unary Scalar Matrix Subtraction", {
    
    has_multiple_gpu_skip()
    
    AS = A[2:4, 2:4]
    C <- -AS
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    
    setContext(1L)
    
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    
    fvclC <- -fvclAS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Single Precision Matrix Addition", {
    
    has_multiple_gpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS + BS
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    setContext(1L)
    
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    fvclBS <- block(fvclB, 2L,4L,2L,4L)
    fvclE <- block(fvclA, 1L,4L,2L,4L)
    
    fvclC <- fvclAS + fvclBS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fvclA + fvclE)
    expect_equal(fvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Single Precision Scalar Matrix Addition", {
    
    has_multiple_gpu_skip()
    
    AS = A[2:4, 2:4]
    C <- AS + 1
    C2 <- 1 + AS
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    
    setContext(1L)
    
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    
    fvclC <- fvclAS + 1
    fvclC2 <- 1 + fvclAS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(fvclC2@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Single Precision Matrix Element-Wise Multiplication", {
    
    has_multiple_gpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS * BS
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    setContext(1L)
    
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    fvclBS <- block(fvclB, 2L,4L,2L,4L)
    
    fvclC <- fvclAS * fvclBS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fvclA * fvclE)
    expect_equal(fvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Single Precision Scalar Matrix Multiplication", {
    
    has_multiple_gpu_skip()
    
    AS = A[2:4, 2:4]
    C <- AS * 2
    C2 <- 2 * AS
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    
    setContext(1L)
    
    fvclC <- fvclAS * 2
    fvclC2 <- 2 * fvclAS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(fvclC2@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Single Precision Matrix Element-Wise Division", {
    
    has_multiple_gpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS / BS
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    setContext(1L)
    
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    fvclBS <- block(fvclB, 2L,4L,2L,4L)
    fvclE <- block(fvclA, 1L,4L,2L,4L)
    
    fvclC <- fvclAS / fvclBS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fvclA / fvclE)
    expect_equal(fvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Single Precision Scalar Matrix Division", {
    
    has_multiple_gpu_skip()
    
    AS = A[2:4, 2:4]
    C <- AS/2
    C2 <- 2/AS
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    
    setContext(1L)
    
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    
    fvclC <- fvclAS/2
    fvclC2 <- 2/fvclAS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(fvclC2@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Single Precision Matrix Element-Wise Power", {
    
    has_multiple_gpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS ^ BS
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    setContext(1L)
    
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    fvclBS <- block(fvclB, 2L,4L,2L,4L)
    fvclE <- block(fvclA, 1L,4L,2L,4L)
    
    fvclC <- fvclAS ^ fvclBS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fvclA ^ fvclE)
    expect_equal(fvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Single Precision Scalar Matrix Power", {
    
    has_multiple_gpu_skip()
    
    AS = A[2:4, 2:4]
    C <- AS^2
    
    setContext(2L)
    
    fvclA <- vclMatrix(A, type="float")
    
    setContext(1L)
    
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    
    fvclC <- fvclAS^2
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(fvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Single Precision crossprod", {
    
    has_multiple_gpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    XS = X[1:2, 2:5]
    YS = Y[1:2, 2:5]
    ZS = Z[2:5, 1:2]
    
    C <- crossprod(XS,YS)
    Cs <- crossprod(XS)
    
    setContext(2L)
    
    fvclX <- vclMatrix(X, type="float")
    fvclY <- vclMatrix(Y, type="float")
    fvclZ <- vclMatrix(Z, type="float")
    
    setContext(1L)
    
    fvclXS <- block(fvclX, 1L,2L,2L,5L)
    fvclYS <- block(fvclY, 1L,2L,2L,5L)
    fvclZS <- block(fvclZ, 2L,5L,1L,2L)
    
    fvclC <- crossprod(fvclXS, fvclYS)
    fvclCs <- crossprod(fvclXS)
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fvclCs[,], Cs, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_error(crossprod(fvclXS, fvclZS))
    expect_equal(fvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(fvclCs@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Single Precision tcrossprod", {
    
    has_multiple_gpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    XS = X[1:2, 2:5]
    YS = Y[1:2, 2:5]
    ZS = Z[2:5, 1:2]
    
    C <- tcrossprod(XS,YS)
    Cs <- tcrossprod(XS)
    
    setContext(2L)
    
    fvclX <- vclMatrix(X, type="float")
    fvclY <- vclMatrix(Y, type="float")
    fvclZ <- vclMatrix(Z, type="float")
    
    setContext(1L)
    
    fvclXS <- block(fvclX, 1L,2L,2L,5L)
    fvclYS <- block(fvclY, 1L,2L,2L,5L)
    fvclZS <- block(fvclZ, 2L,5L,1L,2L)
    
    fvclC <- tcrossprod(fvclXS, fvclYS)
    fvclCs <- tcrossprod(fvclXS)
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fvclCs[,], Cs, tolerance=1e-06, 
                 info="float matrix elements not equivalent") 
    expect_error(crossprod(fvclXS, fvclZS))
    expect_equal(fvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(fvclCs@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

# Double Precision Matrix Block tests

test_that("Switching GPU vclMatrixBlock Double Precision Block Matrix multiplication", {
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS %*% BS
    
    setContext(2L)
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    
    setContext(1L)
    
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    dvclBS <- block(dvclB, 2L,4L,2L,4L)
    
    dvclC <- dvclAS %*% dvclBS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(dvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally") 
})

test_that("Switching GPU vclMatrixBlock Double Precision Matrix Subtraction", {
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS - BS
    
    setContext(2L)
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    
    setContext(1L)
    
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    dvclBS <- block(dvclB, 2L,4L,2L,4L)
    dvclE <- block(dvclA, 1L,4L,2L,4L)
    
    dvclC <- dvclAS - dvclBS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dvclAS - dvclE)
    expect_equal(dvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Double Precision Scalar Matrix Subtraction", {
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    AS = A[2:4, 2:4]
    
    C <- AS - 1
    C2 <- 1 - AS
    
    setContext(2L)
    
    dvclA <- vclMatrix(A, type="double")
    
    setContext(1L)
    
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    
    dvclC <- dvclAS - 1    
    dvclC2 <- 1 - dvclAS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dvclC2, "dvclMatrix")
    expect_equal(dvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(dvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(dvclC2@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Double Precision Unary Scalar Matrix Subtraction", {
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    AS = A[2:4, 2:4]
    C <- -AS
    
    setContext(2L)
    
    dvclA <- vclMatrix(A, type="double")
    
    setContext(1L)
    
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    
    dvclC <- -dvclAS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(dvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Double Precision Matrix Addition", {
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS + BS
    
    setContext(2L)
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    
    setContext(1L)
    
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    dvclBS <- block(dvclB, 2L,4L,2L,4L)
    dvclE <- block(dvclA, 1L,4L,2L,4L)
    
    dvclC <- dvclAS + dvclBS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dvclA + dvclE)
    expect_equal(dvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Double Precision Scalar Matrix Addition", {
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    AS = A[2:4, 2:4]
    C <- AS + 1
    C2 <- 1 + AS
    
    setContext(2L)
    
    dvclA <- vclMatrix(A, type="double")
    
    setContext(1L)
    
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    
    dvclC <- dvclAS + 1
    dvclC2 <- 1 + dvclAS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dvclC2, "dvclMatrix")
    expect_equal(dvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(dvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(dvclC2@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Double Precision Matrix Element-Wise Multiplication", {
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS * BS
    
    setContext(2L)
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    
    setContext(1L)
    
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    dvclBS <- block(dvclB, 2L,4L,2L,4L)
    
    dvclC <- dvclAS * dvclBS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dvclA * dvclE)
    expect_equal(dvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Double Precision Scalar Matrix Multiplication", {
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    AS = A[2:4, 2:4]
    C <- AS * 2
    C2 <- 2 * AS
    
    setContext(2L)
    
    dvclA <- vclMatrix(A, type="double")
    
    setContext(1L)
    
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    
    dvclC <- dvclAS * 2
    dvclC2 <- 2 * dvclAS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dvclC2, "dvclMatrix")
    expect_equal(dvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(dvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(dvclC2@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Double Precision Matrix Element-Wise Division", {
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS / BS
    
    setContext(2L)
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    
    setContext(1L)
    
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    dvclBS <- block(dvclB, 2L,4L,2L,4L)
    dvclE <- block(dvclA, 1L,4L,2L,4L)
    
    dvclC <- dvclAS / dvclBS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dvclA / dvclE)
    expect_equal(dvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Double Precision Scalar Matrix Division", {
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    AS = A[2:4, 2:4]
    C <- AS/2
    C2 <- 2/AS
    
    setContext(2L)
    
    dvclA <- vclMatrix(A, type="double")
    
    setContext(1L)
    
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    
    dvclC <- dvclAS/2
    dvclC2 <- 2/dvclAS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dvclC2, "dvclMatrix")
    expect_equal(dvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(dvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(dvclC2@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Double Precision Matrix Element-Wise Power", {
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS ^ BS
    
    setContext(2L)
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    
    setContext(1L)
    
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    dvclBS <- block(dvclB, 2L,4L,2L,4L)
    dvclE <- block(dvclA, 1L,4L,2L,4L)
    
    dvclC <- dvclAS ^ dvclBS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dvclA ^ dvclE)
    expect_equal(dvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Double Precision Scalar Matrix Power", {
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    AS = A[2:4, 2:4]
    C <- AS^2
    
    setContext(2L)
    
    dvclA <- vclMatrix(A, type="double")
    
    setContext(1L)
    
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    
    dvclC <- dvclAS^2
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_equal(dvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Double Precision crossprod", {
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    XS = X[1:2, 2:5]
    YS = Y[1:2, 2:5]
    ZS = Z[2:5, 1:2]
    
    C <- crossprod(XS,YS)
    Cs <- crossprod(XS)
    
    setContext(2L)
    
    dvclX <- vclMatrix(X, type="double")
    dvclY <- vclMatrix(Y, type="double")
    dvclZ <- vclMatrix(Z, type="double")
    
    setContext(1L)
    
    dvclXS <- block(dvclX, 1L,2L,2L,5L)
    dvclYS <- block(dvclY, 1L,2L,2L,5L)
    dvclZS <- block(dvclZ, 2L,5L,1L,2L)
    
    dvclC <- crossprod(dvclXS, dvclYS)
    dvclCs <- crossprod(dvclXS)
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(dvclCs[,], Cs, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_error(crossprod(dvclXS, dvclZS))
    expect_equal(dvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(dvclCs@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrixBlock Double Precision tcrossprod", {
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    XS = X[1:2, 2:5]
    YS = Y[1:2, 2:5]
    ZS = Z[2:5, 1:2]
    
    C <- tcrossprod(XS,YS)
    Cs <- tcrossprod(XS)
    
    setContext(2L)
    
    dvclX <- vclMatrix(X, type="double")
    dvclY <- vclMatrix(Y, type="double")
    dvclZ <- vclMatrix(Z, type="double")
    
    setContext(1L)
    
    dvclXS <- block(dvclX, 1L,2L,2L,5L)
    dvclYS <- block(dvclY, 1L,2L,2L,5L)
    dvclZS <- block(dvclZ, 2L,5L,1L,2L)
    
    dvclC <- tcrossprod(dvclXS, dvclYS)
    dvclCs <- tcrossprod(dvclXS)
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(dvclCs[,], Cs, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_error(crossprod(dvclXS, dvclZS))
    expect_equal(dvclC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(dvclCs@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})
