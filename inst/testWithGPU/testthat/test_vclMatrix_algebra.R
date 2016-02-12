library(gpuR)
context("vclMatrix algebra")

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
    
    has_gpu_skip()
    
    C <- A %*% B
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    fvclC <- fvclA %*% fvclB
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("vclMatrix Single Precision Matrix Subtraction", {
    
    has_gpu_skip()
    
    C <- A - B
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    fvclC <- fvclA - fvclB
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("vclMatrix Single Precision Scalar Matrix Subtraction", {
    
    has_gpu_skip()
    
    C <- A - 1
    C2 <- 1 - A
    
    fvclA <- vclMatrix(A, type="float")
    
    fvclC <- fvclA - 1    
    fvclC2 <- 1 - fvclA
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("vclMatrix Single Precision Unary Scalar Matrix Subtraction", {
    
    has_gpu_skip()
    
    C <- -A
    
    fvclA <- vclMatrix(A, type="float")
    
    fvclC <- -fvclA
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("vclMatrix Single Precision Matrix Addition", {
    
    has_gpu_skip()
    
    C <- A + B
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    fvclC <- fvclA + fvclB
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("vclMatrix Single Precision Scalar Matrix Addition", {
    
    has_gpu_skip()
    
    C <- A + 1
    C2 <- 1 + A
    
    fvclA <- vclMatrix(A, type="float")
    
    fvclC <- fvclA + 1
    fvclC2 <- 1 + fvclA
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(fvclC2, "fvclMatrix")
    expect_equal(fvclC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("vclMatrix Single Precision Matrix Element-Wise Multiplication", {
    
    has_gpu_skip()
    
    C <- A * B
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    fvclE <- vclMatrix(E, type="float")
    
    fvclC <- fvclA * fvclB
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fvclA * fvclE)
})

test_that("vclMatrix Single Precision Scalar Matrix Multiplication", {
    
    has_gpu_skip()
    
    C <- A * 2
    C2 <- 2 * A
    
    dvclA <- vclMatrix(A, type="float")
    
    dvclC <- dvclA * 2
    dvclC2 <- 2 * dvclA
    
    expect_is(dvclC, "fvclMatrix")
    expect_equal(dvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(dvclC2, "fvclMatrix")
    expect_equal(dvclC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("vclMatrix Single Precision Matrix Element-Wise Division", {
    
    has_gpu_skip()
    
    C <- A / B
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    fvclE <- vclMatrix(E, type="float")
    
    fvclC <- fvclA / fvclB
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fvclA * fvclE)
})

test_that("vclMatrix Single Precision Scalar Matrix Division", {
    
    has_gpu_skip()
    
    C <- A/2
    C2 <- 2/A
    
    dvclA <- vclMatrix(A, type="float")
    
    dvclC <- dvclA/2
    dvclC2 <- 2/dvclA
    
    expect_is(dvclC, "fvclMatrix")
    expect_equal(dvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_is(dvclC2, "fvclMatrix")
    expect_equal(dvclC2[,], C2, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("vclMatrix Single Precision Matrix Element-Wise Power", {
    
    has_gpu_skip()
    
    C <- A ^ B
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    fvclE <- vclMatrix(E, type="float")
    
    fvclC <- fvclA ^ fvclB
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fvclA ^ fvclE)
})

test_that("vclMatrix Single Precision Scalar Matrix Power", {
    
    has_gpu_skip()
    
    C <- A^2
    
    dvclA <- vclMatrix(A, type="float")
    
    dvclC <- dvclA^2
    
    expect_is(dvclC, "fvclMatrix")
    expect_equal(dvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
})

test_that("vclMatrix Single Precision crossprod", {
    
    has_gpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    
    C <- crossprod(X,Y)
    Cs <- crossprod(X)
    
    fvclX <- vclMatrix(X, type="float")
    fvclY <- vclMatrix(Y, type="float")
    fvclZ <- vclMatrix(Z, type="float")
    
    fvclC <- crossprod(fvclX, fvclY)
    fvclCs <- crossprod(fvclX)
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fvclCs[,], Cs, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_error(crossprod(fvclX, fvclZ))
})

test_that("vclMatrix Single Precision tcrossprod", {
    
    has_gpu_skip()
    
    X <- matrix(rnorm(10), nrow=5)
    Y <- matrix(rnorm(10), nrow=5)
    Z <- matrix(rnorm(10), nrow=2)
    
    C <- tcrossprod(X,Y)
    Cs <- tcrossprod(X)
    
    fvclX <- vclMatrix(X, type="float")
    fvclY <- vclMatrix(Y, type="float")
    fvclZ <- vclMatrix(Z, type="float")
    
    fvclC <- tcrossprod(fvclX, fvclY)
    fvclCs <- tcrossprod(fvclX)
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fvclCs[,], Cs, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_error(crossprod(fvclX, fvclZ))
})

# Double Precision Tests

test_that("vclMatrix Double Precision Matrix multiplication", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A %*% B
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    
    dvclC <- dvclA %*% dvclB
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})

test_that("vclMatrix Double Precision Matrix Subtraction", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A - B
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    
    dvclC <- dvclA - dvclB
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})

test_that("vclMatrix Double Precision Matrix Addition", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A + B
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    
    dvclC <- dvclA + dvclB
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})

test_that("vclMatrix Double Precision Scalar Matrix Addition", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A + 1
    C2 <- 1 + A
    
    dvclA <- vclMatrix(A, type="double")
    
    dvclC <- dvclA + 1
    dvclC2 <- 1 + dvclA
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dvclC2, "dvclMatrix")
    expect_equal(dvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("vclMatrix Double Precision Scalar Matrix Subtraction", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A - 1
    C2 <- 1 - A
    
    dvclA <- vclMatrix(A, type="double")
    
    dvclC <- dvclA - 1
    dvclC2 <- 1 - dvclA
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_is(dvclC2, "dvclMatrix")
    expect_equal(dvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("vclMatrix Double Precision Unary Matrix Subtraction", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- -A
    
    fvclA <- vclMatrix(A, type="double")
    
    fvclC <- -fvclA
    
    expect_is(fvclC, "dvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("vclMatrix Double Precision Matrix Element-Wise Multiplication", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A * B
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    dvclE <- vclMatrix(E, type="double")
    
    dvclC <- dvclA * dvclB
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dvclA * dvclE)
})

test_that("vclMatrix Double Precision Scalar Matrix Multiplication", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A * 2
    C2 <- 2 * A
    
    dvclA <- vclMatrix(A, type="double")
    
    dvclC <- dvclA * 2
    dvclC2 <- 2 * dvclA
    
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
    
    C <- A / B
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    dvclE <- vclMatrix(E, type="double")
    
    dvclC <- dvclA / dvclB
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dvclA * dvclE)
})

test_that("vclMatrix Double Precision Scalar Matrix Division", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A/2
    C2 <- 2/A
    
    dvclA <- vclMatrix(A, type="double")
    
    dvclC <- dvclA/2
    dvclC2 <- 2/dvclA
    
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
    
    C <- A ^ B
    
    fvclA <- vclMatrix(A, type="double")
    fvclB <- vclMatrix(B, type="double")
    fvclE <- vclMatrix(E, type="double")
    
    fvclC <- fvclA ^ fvclB
    
    expect_is(fvclC, "dvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(fvclA ^ fvclE)
})

test_that("vclMatrix Double Precision Scalar Matrix Power", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A^2
    
    dvclA <- vclMatrix(A, type="double")
    
    dvclC <- dvclA^2
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
})

test_that("vclMatrix Double Precision crossprod", {
    
    has_gpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    
    C <- crossprod(X,Y)
    Cs <- crossprod(X)
    
    fvclX <- vclMatrix(X, type="double")
    fvclY <- vclMatrix(Y, type="double")
    fvclZ <- vclMatrix(Z, type="double")
    
    fvclC <- crossprod(fvclX, fvclY)
    fvclCs <- crossprod(fvclX)
    
    expect_is(fvclC, "dvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fvclCs[,], Cs, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_error(crossprod(fvclX, fvclZ))
})

test_that("vclMatrix Double Precision tcrossprod", {
    
    has_gpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    
    C <- tcrossprod(X,Y)
    Cs <- tcrossprod(X)
    
    fvclX <- vclMatrix(X, type="double")
    fvclY <- vclMatrix(Y, type="double")
    fvclZ <- vclMatrix(Z, type="double")
    
    fvclC <- tcrossprod(fvclX, fvclY)
    fvclCs <- tcrossprod(fvclX)
    
    expect_is(fvclC, "dvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fvclCs[,], Cs, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_error(crossprod(fvclX, fvclZ))
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
