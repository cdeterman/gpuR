library(gpuR)
context("vclMatrix algebra")

current_context <- set_device_context("gpu")

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
Aint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
Bint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
B <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
E <- matrix(rnorm(15), nrow=5)
v <- rnorm(ORDER^2)

# Single Precision tests

test_that("vclMatrix Single Precision Matrix multiplication", {
    
    has_gpu_skip()
    
    C <- A %*% B
    
    fgpuA <- vclMatrix(A, type="float")
    fgpuB <- vclMatrix(B, type="float")
    fgpuE <- vclMatrix(E, type = "float")
    
    fgpuC <- fgpuA %*% fgpuB
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fgpuA %*% fgpuE,
                 info = "error not thrown for non-conformant matrices")
    
    fgpuC <- A %*% fgpuB
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuC <- fgpuA %*% B
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("vclMatrix Single Precision Matrix Subtraction", {
    
    has_gpu_skip()
    
    C <- A - B
    
    fgpuA <- vclMatrix(A, type="float")
    fgpuB <- vclMatrix(B, type="float")
    fgpuE <- vclMatrix(E, type="float")
    
    fgpuC <- fgpuA - fgpuB
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fgpuA - fgpuE)
    
    fgpuC <- fgpuA - B
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuC <- A - fgpuB
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("vclMatrix Single Precision Scalar Matrix Subtraction", {
    
    has_gpu_skip()
    
    C <- A - 1
    C2 <- 1 - A
    
    fgpuA <- vclMatrix(A, type="float")
    
    fgpuC <- fgpuA - 1    
    fgpuC2 <- 1 - fgpuA
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fgpuC2, "fvclMatrix")
    expect_equal(fgpuC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("vclMatrix Single Precision Unary Scalar Matrix Subtraction", {
    
    has_gpu_skip()
    
    C <- -A
    
    fgpuA <- vclMatrix(A, type="float")
    
    fgpuC <- -fgpuA
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("vclMatrix Single Precision Matrix Addition", {
    
    has_gpu_skip()
    
    C <- A + B
    
    fgpuA <- vclMatrix(A, type="float")
    fgpuB <- vclMatrix(B, type="float")
    fgpuE <- vclMatrix(E, type="float")
    
    fgpuC <- fgpuA + fgpuB
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fgpuA + fgpuE)
    
    fgpuC <- A + fgpuB
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuC <- fgpuA + B
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("vclMatrix Single Precision Scalar Matrix Addition", {
    
    has_gpu_skip()
    
    C <- A + 1
    C2 <- 1 + A
    
    fgpuA <- vclMatrix(A, type="float")
    
    fgpuC <- fgpuA + 1
    fgpuC2 <- 1 + fgpuA
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fgpuC2, "fvclMatrix")
    expect_equal(fgpuC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("vclMatrix Single Precision Matrix Element-Wise Multiplication", {
    
    has_gpu_skip()
    
    C <- A * B
    
    fgpuA <- vclMatrix(A, type="float")
    fgpuB <- vclMatrix(B, type="float")
    fgpuE <- vclMatrix(E, type="float")
    
    fgpuC <- fgpuA * fgpuB
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fgpuA * fgpuE)
    
    fgpuC <- A * fgpuB
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuC <- fgpuA * B
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("vclMatrix Single Precision Scalar Matrix Multiplication", {
    
    has_gpu_skip()
    
    C <- A * 2
    C2 <- 2 * A
    
    dgpuA <- vclMatrix(A, type="float")
    
    dgpuC <- dgpuA * 2
    dgpuC2 <- 2 * dgpuA
    
    expect_is(dgpuC, "fvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(dgpuC2, "fvclMatrix")
    expect_equal(dgpuC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("vclMatrix Single Precision Matrix Element-Wise Division", {
    
    has_gpu_skip()
    
    C <- A / B
    
    fgpuA <- vclMatrix(A, type="float")
    fgpuB <- vclMatrix(B, type="float")
    fgpuE <- vclMatrix(E, type="float")
    
    fgpuC <- fgpuA / fgpuB
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fgpuA / fgpuE)
    
    fgpuC <- A / fgpuB
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuC <- fgpuA / B
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("vclMatrix Single Precision Scalar Matrix Division", {
    
    has_gpu_skip()
    
    C <- A/2
    C2 <- 2/A
    
    dgpuA <- vclMatrix(A, type="float")
    
    dgpuC <- dgpuA/2
    dgpuC2 <- 2/dgpuA
    
    expect_is(dgpuC, "fvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(dgpuC2, "fvclMatrix")
    expect_equal(dgpuC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("vclMatrix Single Precision Matrix Element-Wise Power", {
    
    has_gpu_skip()
    pocl_check()
    
    C <- A ^ B
    
    fgpuA <- vclMatrix(A, type="float")
    fgpuB <- vclMatrix(B, type="float")
    fgpuE <- vclMatrix(E, type="float")
    
    fgpuC <- fgpuA ^ fgpuB
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fgpuA ^ fgpuE)
    
    fgpuC <- A ^ fgpuB
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuC <- fgpuA ^ B
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("vclMatrix Single Precision Scalar Matrix Power", {
    
    has_gpu_skip()
    
    C <- A^2
    C2 <- 2^A
    
    dgpuA <- vclMatrix(A, type="float")
    
    dgpuC <- dgpuA^2
    dgpuC2 <- 2^dgpuA
    
    expect_is(dgpuC, "fvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_equal(dgpuC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("vclMatrix Single Precision crossprod", {
    
    has_gpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    
    C <- crossprod(X,Y)
    Cs <- crossprod(X)
    
    fgpuX <- vclMatrix(X, type="float")
    fgpuY <- vclMatrix(Y, type="float")
    fgpuZ <- vclMatrix(Z, type="float")
    
    fgpuC <- crossprod(fgpuX, fgpuY)
    fgpuCs <- crossprod(fgpuX)
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fgpuCs[,], Cs, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_error(crossprod(fgpuX, fgpuZ))
    
    fgpuC <- crossprod(fgpuX, Y)
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuC <- crossprod(X, fgpuY)
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("vclMatrix Single Precision tcrossprod", {
    
    has_gpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(12), nrow=2)
    
    C <- tcrossprod(X,Y)
    Cs <- tcrossprod(X)
    
    fgpuX <- vclMatrix(X, type="float")
    fgpuY <- vclMatrix(Y, type="float")
    fgpuZ <- vclMatrix(Z, type="float")
    
    fgpuC <- tcrossprod(fgpuX, fgpuY)
    fgpuCs <- tcrossprod(fgpuX)
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fgpuCs[,], Cs, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_error(tcrossprod(fgpuX, fgpuZ))
    
    fgpuC <- tcrossprod(fgpuX, Y)
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuC <- tcrossprod(X, fgpuY)
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("vclMatrix Single Precision transpose", {
    
    has_gpu_skip()
    
    At <- t(A)
    
    fgpuA <- vclMatrix(A, type="float")
    fgpuAt <- t(fgpuA)
    
    expect_is(fgpuAt, "fvclMatrix")
    expect_equal(fgpuAt[,], At, tolerance=1e-07, 
                 info="transposed float matrix elements not equivalent") 
})

test_that("vclMatrix Single Precision determinant", {
    
    has_gpu_skip()
    
    d <- det(A)
    
    fgpuA <- vclMatrix(A, type="float")
    fgpud <- det(fgpuA)
    
    expect_is(fgpud, "numeric")
    expect_equal(fgpud, d, tolerance=1e-07, 
                 info="float determinants not equivalent") 
})

# Integer tests

test_that("vclMatrix Integer Matrix multiplication", {
    
    has_gpu_skip()
    
    Cint <- Aint %*% Bint
    
    igpuA <- vclMatrix(Aint, type="integer")
    igpuB <- vclMatrix(Bint, type="integer")
    
    igpuC <- igpuA %*% igpuB
    
    expect_equivalent(igpuC[,], Cint,
                      info="integer matrix elements not equivalent")
    
    igpuC <- Aint %*% igpuB
    
    expect_equivalent(igpuC[,], Cint,
                      info="integer matrix elements not equivalent")
    
    igpuC <- igpuA %*% Bint
    
    expect_equivalent(igpuC[,], Cint,
                      info="integer matrix elements not equivalent")
})

test_that("vclMatrix Integer Matrix Subtraction", {
    
    has_gpu_skip()
    
    Cint <- Aint - Bint
    
    igpuA <- vclMatrix(Aint, type="integer")
    igpuB <- vclMatrix(Bint, type="integer")
    
    igpuC <- igpuA - igpuB
    
    expect_is(igpuC, "ivclMatrix")
    expect_equal(igpuC[,], Cint,
                 info="integer matrix elements not equivalent")
    
    
    igpuC <- igpuA - Bint
    
    expect_is(igpuC, "ivclMatrix")
    expect_equal(igpuC[,], Cint,
                 info="integer matrix elements not equivalent")
    
    
    igpuC <- Aint - igpuB
    
    expect_is(igpuC, "ivclMatrix")
    expect_equal(igpuC[,], Cint,
                 info="integer matrix elements not equivalent")
})

test_that("vclMatrix Integer Precision Scalar Matrix Subtraction", {
    
    has_gpu_skip()
    
    C <- Aint - 1L
    C2 <- 1L - Aint
    
    fgpuA <- vclMatrix(Aint, type="integer")
    
    fgpuC <- fgpuA - 1L
    fgpuC2 <- 1L - fgpuA
    
    expect_is(fgpuC, "ivclMatrix")
    expect_equal(fgpuC[,], C, 
                 info="integer matrix elements not equivalent") 
    expect_is(fgpuC2, "ivclMatrix")
    expect_equal(fgpuC2[,], C2,
                 info="intger matrix elements not equivalent") 
})

test_that("vclMatrix Integer Precision Unary Scalar Matrix Subtraction", {
    
    has_gpu_skip()
    
    C <- -Aint
    
    fgpuA <- vclMatrix(Aint, type="integer")
    
    fgpuC <- -fgpuA
    
    expect_is(fgpuC, "ivclMatrix")
    expect_equal(fgpuC[,], C,
                 info="integer matrix elements not equivalent") 
})

test_that("vclMatrix Integer Matrix Addition", {
    
    has_gpu_skip()
    
    Cint <- Aint + Bint
    
    igpuA <- vclMatrix(Aint, type="integer")
    igpuB <- vclMatrix(Bint, type="integer")
    
    igpuC <- igpuA + igpuB
    
    expect_is(igpuC, "ivclMatrix")
    expect_equal(igpuC[,], Cint,
                 info="integer matrix elements not equivalent")
    
    igpuC <- Aint + igpuB
    
    expect_is(igpuC, "ivclMatrix")
    expect_equal(igpuC[,], Cint,
                 info="integer matrix elements not equivalent")
    
    igpuC <- igpuA + Bint
    
    expect_is(igpuC, "ivclMatrix")
    expect_equal(igpuC[,], Cint,
                 info="integer matrix elements not equivalent")
})

test_that("vclMatrix Integer Precision Scalar Matrix Addition", {
    
    has_gpu_skip()
    
    C <- Aint + 1L
    C2 <- 1L + Aint
    
    fgpuA <- vclMatrix(Aint, type="integer")
    
    fgpuC <- fgpuA + 1L
    fgpuC2 <- 1L + fgpuA
    
    expect_is(fgpuC, "ivclMatrix")
    expect_equal(fgpuC[,], C,
                 info="integer matrix elements not equivalent") 
    expect_is(fgpuC2, "ivclMatrix")
    expect_equal(fgpuC2[,], C2,
                 info="integer matrix elements not equivalent") 
})

test_that("vclMatrix Integer Precision Matrix Element-Wise Multiplication", {
    
    has_gpu_skip()
    
    C <- Aint * Bint
    
    fgpuA <- vclMatrix(Aint, type="integer")
    fgpuB <- vclMatrix(Bint, type="integer")
    
    fgpuC <- fgpuA * fgpuB
    
    expect_is(fgpuC, "ivclMatrix")
    expect_equal(fgpuC[,], C, 
                 info="integer matrix elements not equivalent") 
    
    fgpuC <- Aint * fgpuB
    
    expect_is(fgpuC, "ivclMatrix")
    expect_equal(fgpuC[,], C, 
                 info="integer matrix elements not equivalent") 
    
    fgpuC <- fgpuA * Bint
    
    expect_is(fgpuC, "ivclMatrix")
    expect_equal(fgpuC[,], C, 
                 info="integer matrix elements not equivalent") 
})

test_that("vclMatrix Integer Precision Scalar Matrix Multiplication", {
    
    has_gpu_skip()
    
    C <- Aint * 2L
    C2 <- 2L * Aint
    
    dgpuA <- vclMatrix(Aint, type="integer")
    
    dgpuC <- dgpuA * 2L
    dgpuC2 <- 2L * dgpuA
    
    expect_is(dgpuC, "ivclMatrix")
    expect_equal(dgpuC[,], C,
                 info="integer matrix elements not equivalent") 
    expect_is(dgpuC2, "ivclMatrix")
    expect_equal(dgpuC2[,], C2,
                 info="integer matrix elements not equivalent") 
})

test_that("vclMatrix Integer Precision Matrix Element-Wise Division", {
    
    has_gpu_skip()
    
    C <- Aint / Bint
    C <- apply(C, 2, as.integer)
    
    fgpuA <- vclMatrix(Aint, type="integer")
    fgpuB <- vclMatrix(Bint, type="integer")
    
    fgpuC <- fgpuA / fgpuB
    
    expect_is(fgpuC, "ivclMatrix")
    expect_equal(fgpuC[,], C,
                 info="integer matrix elements not equivalent") 
    
    fgpuC <- Aint / fgpuB
    
    expect_is(fgpuC, "ivclMatrix")
    expect_equal(fgpuC[,], C,
                 info="integer matrix elements not equivalent") 
    
    fgpuC <- fgpuA / Bint
    
    expect_is(fgpuC, "ivclMatrix")
    expect_equal(fgpuC[,], C,
                 info="integer matrix elements not equivalent") 
})

test_that("vclMatrix Integer Precision Scalar Matrix Division", {
    
    has_gpu_skip()
    
    C <- Aint/2L
    C2 <- 2L/Aint
    
    C <- apply(C, 2, as.integer)
    C2 <- apply(C2, 2, as.integer)
    
    dgpuA <- vclMatrix(Aint, type="integer")
    
    dgpuC <- dgpuA/2L
    dgpuC2 <- 2L/dgpuA
    
    expect_is(dgpuC, "ivclMatrix")
    expect_equal(dgpuC[,], C,
                 info="integer matrix elements not equivalent") 
    expect_is(dgpuC2, "ivclMatrix")
    expect_equal(dgpuC2[,], C2,
                 info="integer matrix elements not equivalent") 
})

test_that("vclMatrix Integer Precision Matrix Element-Wise Power", {
    
    has_gpu_skip()
    pocl_check()
    
    Apow <- matrix(seq.int(9), ncol=3, nrow=3)
    Bpow <- matrix(2, ncol = 3, nrow = 3)
    C <- Apow ^ Bpow
    
    fgpuA <- vclMatrix(Apow, type="integer")
    fgpuB <- vclMatrix(Bpow, type="integer")
    
    fgpuC <- fgpuA ^ fgpuB
    
    expect_is(fgpuC, "ivclMatrix")
    expect_equal(fgpuC[,], C,
                 info="integer matrix elements not equivalent")
    
    fgpuC <- Apow ^ fgpuB
    
    expect_is(fgpuC, "ivclMatrix")
    expect_equal(fgpuC[,], C,
                 info="integer matrix elements not equivalent")
    
    fgpuC <- fgpuA ^ Bpow
    
    expect_is(fgpuC, "ivclMatrix")
    expect_equal(fgpuC[,], C,
                 info="integer matrix elements not equivalent")
})

test_that("vclMatrix Integer Precision Scalar Matrix Power", {
    
    has_gpu_skip()
    
    C <- Aint^2L
    C2 <- 2L^Aint
    
    C <- apply(C, 2, as.integer)
    C2 <- apply(C2, 2, as.integer)
    
    dgpuA <- vclMatrix(Aint, type="integer")
    
    dgpuC <- dgpuA^2L
    dgpuC2 <- 2L^dgpuA
    
    expect_is(dgpuC, "ivclMatrix")
    expect_equal(dgpuC[,], C,
                 info="integer matrix elements not equivalent")
    expect_equal(dgpuC2[,], C2,
                 info="integer matrix elements not equivalent")
})

# Double Precision tests

test_that("vclMatrix Double Precision Matrix multiplication", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A %*% B
    
    dgpuA <- vclMatrix(A, type="double")
    dgpuB <- vclMatrix(B, type="double")
    
    dgpuC <- dgpuA %*% dgpuB
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    
    dgpuC <- dgpuA %*% B
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    
    dgpuC <- A %*% dgpuB
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})

test_that("vclMatrix Double Precision Matrix Subtraction", {
    
    has_gpu_skip()
    has_double_skip()
    
    
    C <- A - B
    
    dgpuA <- vclMatrix(A, type="double")
    dgpuB <- vclMatrix(B, type="double")
    dgpuE <- vclMatrix(E, type="double")
    
    dgpuC <- dgpuA - dgpuB
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dgpuA - dgpuE)
    
    dgpuC <- A - dgpuB
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    
    dgpuC <- dgpuA - B
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})

test_that("vclMatrix Double Precision Matrix Addition", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A + B
    
    dgpuA <- vclMatrix(A, type="double")
    dgpuB <- vclMatrix(B, type="double")
    dgpuE <- vclMatrix(E, type="double")
    
    dgpuC <- dgpuA + dgpuB
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dgpuA + dgpuE)
    
    dgpuC <- A + dgpuB
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    
    dgpuC <- dgpuA + B
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})

test_that("vclMatrix Double Precision Scalar Matrix Addition", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A + 1
    C2 <- 1 + A
    
    dgpuA <- vclMatrix(A, type="double")
    
    dgpuC <- dgpuA + 1
    dgpuC2 <- 1 + dgpuA
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dgpuC2, "dvclMatrix")
    expect_equal(dgpuC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("vclMatrix Double Precision Scalar Matrix Subtraction", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A - 1
    C2 <- 1 - A
    
    dgpuA <- vclMatrix(A, type="double")
    
    dgpuC <- dgpuA - 1
    dgpuC2 <- 1 - dgpuA
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dgpuC2, "dvclMatrix")
    expect_equal(dgpuC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("vclMatrix Double Precision Unary Matrix Subtraction", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- -A
    
    fgpuA <- vclMatrix(A, type="double")
    
    fgpuC <- -fgpuA
    
    expect_is(fgpuC, "dvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("vclMatrix Double Precision Matrix Element-Wise Multiplication", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A * B
    
    dgpuA <- vclMatrix(A, type="double")
    dgpuB <- vclMatrix(B, type="double")
    dgpuE <- vclMatrix(E, type="double")
    
    dgpuC <- dgpuA * dgpuB
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dgpuA * dgpuE)
    
    dgpuC <- A * dgpuB
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    
    dgpuC <- dgpuA * B
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("vclMatrix Double Precision Scalar Matrix Multiplication", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A * 2
    C2 <- 2 * A
    
    dgpuA <- vclMatrix(A, type="double")
    
    dgpuC <- dgpuA * 2
    dgpuC2 <- 2 * dgpuA
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dgpuC2, "dvclMatrix")
    expect_equal(dgpuC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("vclMatrix Double Precision Matrix Element-Wise Division", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A / B
    
    dgpuA <- vclMatrix(A, type="double")
    dgpuB <- vclMatrix(B, type="double")
    dgpuE <- vclMatrix(E, type="double")
    
    dgpuC <- dgpuA / dgpuB
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dgpuA * dgpuE)
    
    dgpuC <- A / dgpuB
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    
    dgpuC <- dgpuA / B
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})

test_that("vclMatrix Double Precision Scalar Matrix Division", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A/2
    C2 <- 2/A
    
    dgpuA <- vclMatrix(A, type="double")
    
    dgpuC <- dgpuA/2
    dgpuC2 <- 2/dgpuA
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dgpuC2, "dvclMatrix")
    expect_equal(dgpuC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("vclMatrix Double Precision Matrix Element-Wise Power", {
    
    has_gpu_skip()
    has_double_skip()
    pocl_check()
    
    C <- A ^ B
    
    fgpuA <- vclMatrix(A, type="double")
    fgpuB <- vclMatrix(B, type="double")
    fgpuE <- vclMatrix(E, type="double")
    
    fgpuC <- fgpuA ^ fgpuB
    
    expect_is(fgpuC, "dvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(fgpuA ^ fgpuE)
    
    fgpuC <- A ^ fgpuB
    
    expect_is(fgpuC, "dvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    
    fgpuC <- fgpuA ^ B
    
    expect_is(fgpuC, "dvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})

test_that("vclMatrix Double Precision Scalar Matrix Power", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A^2
    
    dgpuA <- vclMatrix(A, type="double")
    
    dgpuC <- dgpuA^2
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("vclMatrix Double Precision crossprod", {
    
    has_gpu_skip()
    has_double_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    
    C <- crossprod(X,Y)
    Cs <- crossprod(X)
    
    fgpuX <- vclMatrix(X, type="double")
    fgpuY <- vclMatrix(Y, type="double")
    fgpuZ <- vclMatrix(Z, type="double")
    
    fgpuC <- crossprod(fgpuX, fgpuY)
    fgpuCs <- crossprod(fgpuX)
    
    expect_is(fgpuC, "dvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fgpuCs[,], Cs, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_error(crossprod(fgpuX, fgpuZ))
    
    fgpuC <- crossprod(fgpuX, Y)
    
    expect_is(fgpuC, "dvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    
    fgpuC <- crossprod(X, fgpuY)
    
    expect_is(fgpuC, "dvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})

test_that("vclMatrix Double Precision tcrossprod", {
    
    has_gpu_skip()
    has_double_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(12), nrow=2)
    
    C <- tcrossprod(X,Y)
    Cs <- tcrossprod(X)
    
    fgpuX <- vclMatrix(X, type="double")
    fgpuY <- vclMatrix(Y, type="double")
    fgpuZ <- vclMatrix(Z, type="double")
    
    fgpuC <- tcrossprod(fgpuX, fgpuY)
    fgpuCs <- tcrossprod(fgpuX)
    
    expect_is(fgpuC, "dvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fgpuCs[,], Cs, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_error(tcrossprod(fgpuX, fgpuZ))
    
    fgpuC <- tcrossprod(fgpuX, Y)
    
    expect_is(fgpuC, "dvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    
    fgpuC <- tcrossprod(X, fgpuY)
    
    expect_is(fgpuC, "dvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})

test_that("vclMatrix Double Precision transpose", {

    has_gpu_skip()
    has_double_skip()

    At <- t(A)

    fgpuA <- vclMatrix(A, type="double")
    fgpuAt <- t(fgpuA)

    expect_is(fgpuAt, "dvclMatrix")
    expect_equal(fgpuAt[,], At, tolerance=.Machine$double.eps^0.5,
                 info="transposed double matrix elements not equivalent")
})


# test_that("vclMatrix Integer Matrix multiplication", {
#
#     has_gpu_skip()
#
#     Cint <- Aint %*% Bint
#
#     igpuA <- vclMatrix(Aint, type="integer")
#     igpuB <- vclMatrix(Bint, type="integer")
#
#     igpuC <- igpuA %*% igpuB
#
#     expect_equivalent(igpuC[,], Cint,
#                       info="float matrix elements not equivalent")
# })
#
# test_that("vclMatrix Integer Matrix Subtraction", {
#
#     has_gpu_skip()
#
#     Cint <- Aint - Bint
#
#     igpuA <- vclMatrix(Aint, type="integer")
#     igpuB <- vclMatrix(Bint, type="integer")
#
#     igpuC <- igpuA - igpuB
#
#     expect_is(igpuC, "ivclMatrix")
#     expect_equal(igpuC[,], Cint,
#                  info="integer matrix elements not equivalent")
# })
#
# test_that("vclMatrix Integer Matrix Addition", {
#
#     has_gpu_skip()
#
#     Cint <- Aint + Bint
#
#     igpuA <- vclMatrix(Aint, type="integer")
#     igpuB <- vclMatrix(Bint, type="integer")
#
#     igpuC <- igpuA + igpuB
#
#     expect_is(igpuC, "ivclMatrix")
#     expect_equal(igpuC[,], Cint,
#                  info="integer matrix elements not equivalent")
# })



test_that("CPU vclMatrix Diagonal access", {

    has_gpu_skip()

    fgpuA <- vclMatrix(A, type="float")

    D <- diag(A)
    gpuD <- diag(fgpuA)

    expect_is(gpuD, "fvclVector")
    expect_equal(gpuD[,], D, tolerance=1e-07,
                 info="float matrix diagonal elements not equivalent")

    vec <- rnorm(ORDER)
    diag(fgpuA) <- vclVector(vec, type = "float")
    diag(A) <- vec

    expect_equal(fgpuA[,], A, tolerance=1e-07,
                 info="set float matrix diagonal elements not equivalent")

    has_double_skip()

    fgpuA <- vclMatrix(A, type="double")

    D <- diag(A)
    gpuD <- diag(fgpuA)

    expect_is(gpuD, "dvclVector")
    expect_equal(gpuD[,], D, tolerance=.Machine$double.eps^0.5,
                 info="double matrix diagonal elements not equivalent")

    vec <- rnorm(ORDER)
    diag(fgpuA) <- vclVector(vec, type = "double")
    diag(A) <- vec

    expect_equal(fgpuA[,], A, tolerance=.Machine$double.eps^0.5,
                 info="set double matrix diagonal elements not equivalent")
})

test_that("vclMatrix Double Precision determinant", {
    
    has_gpu_skip()
    has_double_skip()
    
    d <- det(A)
    
    fgpuA <- vclMatrix(A, type="double")
    fgpud <- det(fgpuA)
    
    expect_is(fgpud, "numeric")
    expect_equal(fgpud, d, tolerance=.Machine$double.eps^0.5, 
                 info="double determinants not equivalent") 
})

setContext(current_context)
