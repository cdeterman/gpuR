library(gpuR)
context("vclMatrixBlock algebra")

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

test_that("vclMatrix Single Precision Block Matrix multiplication", {
    
    has_gpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS %*% BS
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    fvclBS <- block(fvclB, 2L,4L,2L,4L)
    
    fvclC <- fvclAS %*% fvclBS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("vclMatrix Single Precision Matrix Subtraction", {
    
    has_gpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS - BS
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    fvclBS <- block(fvclB, 2L,4L,2L,4L)
    fvclE <- block(fvclA, 1L,4L,2L,4L)
    
    fvclC <- fvclAS - fvclBS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fvclAS - fvclE)
})

test_that("vclMatrix Single Precision Scalar Matrix Subtraction", {
    
    has_gpu_skip()
    
    AS = A[2:4, 2:4]
    
    C <- AS - 1
    C2 <- 1 - AS
    
    fvclA <- vclMatrix(A, type="float")
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    
    fvclC <- fvclAS - 1    
    fvclC2 <- 1 - fvclAS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("vclMatrix Single Precision Unary Scalar Matrix Subtraction", {
    
    has_gpu_skip()
    
    AS = A[2:4, 2:4]
    C <- -AS
    
    fvclA <- vclMatrix(A, type="float")
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    
    fvclC <- -fvclAS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("vclMatrix Single Precision Matrix Addition", {
    
    has_gpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS + BS
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    fvclBS <- block(fvclB, 2L,4L,2L,4L)
    fvclE <- block(fvclA, 1L,4L,2L,4L)
    
    fvclC <- fvclAS + fvclBS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fvclA + fvclE)
})

test_that("vclMatrix Single Precision Scalar Matrix Addition", {
    
    has_gpu_skip()
    
    AS = A[2:4, 2:4]
    C <- AS + 1
    C2 <- 1 + AS
    
    fvclA <- vclMatrix(A, type="float")
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    
    fvclC <- fvclAS + 1
    fvclC2 <- 1 + fvclAS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("vclMatrix Single Precision Matrix Element-Wise Multiplication", {
    
    has_gpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS * BS
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    fvclBS <- block(fvclB, 2L,4L,2L,4L)
    
    fvclC <- fvclAS * fvclBS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fvclA * fvclE)
})

test_that("vclMatrix Single Precision Scalar Matrix Multiplication", {
    
    has_gpu_skip()
    
    AS = A[2:4, 2:4]
    C <- AS * 2
    C2 <- 2 * AS
    
    fvclA <- vclMatrix(A, type="float")
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    
    fvclC <- fvclAS * 2
    fvclC2 <- 2 * fvclAS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("vclMatrix Single Precision Matrix Element-Wise Division", {
    
    has_gpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS / BS
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    fvclBS <- block(fvclB, 2L,4L,2L,4L)
    fvclE <- block(fvclA, 1L,4L,2L,4L)
    
    fvclC <- fvclAS / fvclBS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fvclA / fvclE)
})

test_that("vclMatrix Single Precision Scalar Matrix Division", {
    
    has_gpu_skip()
    
    AS = A[2:4, 2:4]
    C <- AS/2
    C2 <- 2/AS
    
    fvclA <- vclMatrix(A, type="float")
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    
    fvclC <- fvclAS/2
    fvclC2 <- 2/fvclAS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("vclMatrix Single Precision Matrix Element-Wise Power", {
    
    has_gpu_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS ^ BS
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    fvclBS <- block(fvclB, 2L,4L,2L,4L)
    fvclE <- block(fvclA, 1L,4L,2L,4L)
    
    fvclC <- fvclAS ^ fvclBS
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fvclA ^ fvclE)
})

test_that("vclMatrix Single Precision Scalar Matrix Power", {
    
    has_gpu_skip()
    
    AS = A[2:4, 2:4]
    C <- AS^2
    
    fvclA <- vclMatrix(A, type="float")
    fvclAS <- block(fvclA, 2L,4L,2L,4L)
    
    fvclC <- fvclAS^2
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("vclMatrix Single Precision crossprod", {
    
    has_gpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    XS = X[1:2, 2:5]
    YS = Y[1:2, 2:5]
    ZS = Z[2:5, 1:2]
    
    C <- crossprod(XS,YS)
    Cs <- crossprod(XS)
    
    fvclX <- vclMatrix(X, type="float")
    fvclY <- vclMatrix(Y, type="float")
    fvclZ <- vclMatrix(Z, type="float")
    
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
})

test_that("vclMatrix Single Precision tcrossprod", {
    
    has_gpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    XS = X[1:2, 2:5]
    YS = Y[1:2, 2:5]
    ZS = Z[2:5, 1:2]
    
    C <- tcrossprod(XS,YS)
    Cs <- tcrossprod(XS)
    
    fvclX <- vclMatrix(X, type="float")
    fvclY <- vclMatrix(Y, type="float")
    fvclZ <- vclMatrix(Z, type="float")
    
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
})

# Double Precision Matrix Block tests

test_that("vclMatrix Double Precision Block Matrix multiplication", {
    
    has_gpu_skip()
    has_double_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS %*% BS
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    dvclBS <- block(dvclB, 2L,4L,2L,4L)
    
    dvclC <- dvclAS %*% dvclBS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})

test_that("vclMatrix Double Precision Matrix Subtraction", {
    
    has_gpu_skip()
    has_double_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS - BS
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    dvclBS <- block(dvclB, 2L,4L,2L,4L)
    dvclE <- block(dvclA, 1L,4L,2L,4L)
    
    dvclC <- dvclAS - dvclBS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dvclAS - dvclE)
})

test_that("vclMatrix Double Precision Scalar Matrix Subtraction", {
    
    has_gpu_skip()
    has_double_skip()
    
    AS = A[2:4, 2:4]
    
    C <- AS - 1
    C2 <- 1 - AS
    
    dvclA <- vclMatrix(A, type="double")
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    
    dvclC <- dvclAS - 1    
    dvclC2 <- 1 - dvclAS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dvclC2, "dvclMatrix")
    expect_equal(dvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("vclMatrix Double Precision Unary Scalar Matrix Subtraction", {
    
    has_gpu_skip()
    has_double_skip()
    
    AS = A[2:4, 2:4]
    C <- -AS
    
    dvclA <- vclMatrix(A, type="double")
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    
    dvclC <- -dvclAS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("vclMatrix Double Precision Matrix Addition", {
    
    has_gpu_skip()
    has_double_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS + BS
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    dvclBS <- block(dvclB, 2L,4L,2L,4L)
    dvclE <- block(dvclA, 1L,4L,2L,4L)
    
    dvclC <- dvclAS + dvclBS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dvclA + dvclE)
})

test_that("vclMatrix Double Precision Scalar Matrix Addition", {
    
    has_gpu_skip()
    has_double_skip()
    
    AS = A[2:4, 2:4]
    C <- AS + 1
    C2 <- 1 + AS
    
    dvclA <- vclMatrix(A, type="double")
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    
    dvclC <- dvclAS + 1
    dvclC2 <- 1 + dvclAS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dvclC2, "dvclMatrix")
    expect_equal(dvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("vclMatrix Double Precision Matrix Element-Wise Multiplication", {
    
    has_gpu_skip()
    has_double_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS * BS
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    dvclBS <- block(dvclB, 2L,4L,2L,4L)
    
    dvclC <- dvclAS * dvclBS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dvclA * dvclE)
})

test_that("vclMatrix Double Precision Scalar Matrix Multiplication", {
    
    has_gpu_skip()
    has_double_skip()
    
    AS = A[2:4, 2:4]
    C <- AS * 2
    C2 <- 2 * AS
    
    dvclA <- vclMatrix(A, type="double")
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    
    dvclC <- dvclAS * 2
    dvclC2 <- 2 * dvclAS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dvclC2, "dvclMatrix")
    expect_equal(dvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("vclMatrix Double Precision Matrix Element-Wise Division", {
    
    has_gpu_skip()
    has_double_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS / BS
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    dvclBS <- block(dvclB, 2L,4L,2L,4L)
    dvclE <- block(dvclA, 1L,4L,2L,4L)
    
    dvclC <- dvclAS / dvclBS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dvclA / dvclE)
})

test_that("vclMatrix Double Precision Scalar Matrix Division", {
    
    has_gpu_skip()
    has_double_skip()
    
    AS = A[2:4, 2:4]
    C <- AS/2
    C2 <- 2/AS
    
    dvclA <- vclMatrix(A, type="double")
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    
    dvclC <- dvclAS/2
    dvclC2 <- 2/dvclAS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dvclC2, "dvclMatrix")
    expect_equal(dvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("vclMatrix Double Precision Matrix Element-Wise Power", {
    
    has_gpu_skip()
    has_double_skip()
    
    AS = A[2:4, 2:4]
    BS = B[2:4, 2:4]
    C <- AS ^ BS
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    dvclBS <- block(dvclB, 2L,4L,2L,4L)
    dvclE <- block(dvclA, 1L,4L,2L,4L)
    
    dvclC <- dvclAS ^ dvclBS
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dvclA ^ dvclE)
})

test_that("vclMatrix Double Precision Scalar Matrix Power", {
    
    has_gpu_skip()
    has_double_skip()
    
    AS = A[2:4, 2:4]
    C <- AS^2
    
    dvclA <- vclMatrix(A, type="double")
    dvclAS <- block(dvclA, 2L,4L,2L,4L)
    
    dvclC <- dvclAS^2
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("vclMatrix Double Precision crossprod", {
    
    has_gpu_skip()
    has_double_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    XS = X[1:2, 2:5]
    YS = Y[1:2, 2:5]
    ZS = Z[2:5, 1:2]
    
    C <- crossprod(XS,YS)
    Cs <- crossprod(XS)
    
    dvclX <- vclMatrix(X, type="double")
    dvclY <- vclMatrix(Y, type="double")
    dvclZ <- vclMatrix(Z, type="double")
    
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
})

test_that("vclMatrix Double Precision tcrossprod", {
    
    has_gpu_skip()
    has_double_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    XS = X[1:2, 2:5]
    YS = Y[1:2, 2:5]
    ZS = Z[2:5, 1:2]
    
    C <- tcrossprod(XS,YS)
    Cs <- tcrossprod(XS)
    
    dvclX <- vclMatrix(X, type="double")
    dvclY <- vclMatrix(Y, type="double")
    dvclZ <- vclMatrix(Z, type="double")
    
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
})
