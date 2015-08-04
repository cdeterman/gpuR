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


test_that("vclMatrix Single Precision Matrix multiplication successful", {
    
    has_gpu_skip()
    
    C <- A %*% B
    
    fgpuA <- vclMatrix(A, type="float")
    fgpuB <- vclMatrix(B, type="float")
    
    fgpuC <- fgpuA %*% fgpuB
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("vclMatrix Double Precision Matrix multiplication successful", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A %*% B
    
    dgpuA <- vclMatrix(A, type="double")
    dgpuB <- vclMatrix(B, type="double")
    
    dgpuC <- dgpuA %*% dgpuB
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})

test_that("vclMatrix Single Precision Matrix Subtraction successful", {
    
    has_gpu_skip()
    
    C <- A - B
    
    fgpuA <- vclMatrix(A, type="float")
    fgpuB <- vclMatrix(B, type="float")
    
    fgpuC <- fgpuA - fgpuB
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("vclMatrix Single Precision Matrix Addition successful", {
    
    has_gpu_skip()
    
    C <- A + B
    
    fgpuA <- vclMatrix(A, type="float")
    fgpuB <- vclMatrix(B, type="float")
    
    fgpuC <- fgpuA + fgpuB
    
    expect_is(fgpuC, "fvclMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("vclMatrix Double Precision Matrix Subtraction successful", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A - B
    
    dgpuA <- vclMatrix(A, type="double")
    dgpuB <- vclMatrix(B, type="double")
    
    dgpuC <- dgpuA - dgpuB
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})

test_that("vclMatrix Double Precision Matrix Addition successful", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A + B
    
    dgpuA <- vclMatrix(A, type="double")
    dgpuB <- vclMatrix(B, type="double")
    
    dgpuC <- dgpuA + dgpuB
    
    expect_is(dgpuC, "dvclMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})


test_that("vclMatrix Single Precision crossprod successful", {
    
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
})


test_that("vclMatrix Double Precision crossprod successful", {
    
    has_gpu_skip()
    
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
})


test_that("vclMatrix Single Precision tcrossprod successful", {
    
    has_gpu_skip()
    
    X <- matrix(rnorm(10), nrow=5)
    Y <- matrix(rnorm(10), nrow=5)
    Z <- matrix(rnorm(10), nrow=2)
    
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
    expect_error(crossprod(fgpuX, fgpuZ))
})


test_that("vclMatrix Double Precision tcrossprod successful", {
    
    has_gpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    
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
    expect_error(crossprod(fgpuX, fgpuZ))
})





# test_that("vclMatrix Integer Matrix multiplication successful", {
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
# test_that("vclMatrix Integer Matrix Subtraction successful", {
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
# test_that("vclMatrix Integer Matrix Addition successful", {
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
