library(gpuR)
context("gpuMatrix algebra")

# avoid downcast warnings for single precision
options(bigmemory.typecast.warning=FALSE)

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
Aint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
Bint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
B <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)


test_that("gpuMatrix Integer Matrix multiplication successful", {
    
    has_gpu_skip()
    
    Cint <- Aint %*% Bint
    
    igpuA <- gpuMatrix(Aint, type="integer")
    igpuB <- gpuMatrix(Bint, type="integer")
    
    igpuC <- igpuA %*% igpuB
    
    expect_equivalent(igpuC[,], Cint, 
                      info="float matrix elements not equivalent")      
})

test_that("gpuMatrix Single Precision Matrix multiplication successful", {
    
    has_gpu_skip()
    
    C <- A %*% B
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    
    fgpuC <- fgpuA %*% fgpuB
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("gpuMatrix Single Precision Matrix Subtraction successful", {
    
    has_gpu_skip()
    
    C <- A - B
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")

    fgpuC <- fgpuA - fgpuB
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("gpuMatrix Single Precision Matrix Addition successful", {
    
    has_gpu_skip()
    
    C <- A + B
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    
    fgpuC <- fgpuA + fgpuB
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("gpuMatrix Integer Matrix Subtraction successful", {
    
    has_gpu_skip()
    
    Cint <- Aint - Bint
    
    igpuA <- gpuMatrix(Aint, type="integer")
    igpuB <- gpuMatrix(Bint, type="integer")
    
    igpuC <- igpuA - igpuB
    
    expect_is(igpuC, "igpuMatrix")
    expect_equal(igpuC[,], Cint, 
                 info="integer matrix elements not equivalent")  
})

test_that("gpuMatrix Integer Matrix Addition successful", {
    
    has_gpu_skip()
    
    Cint <- Aint + Bint
    
    igpuA <- gpuMatrix(Aint, type="integer")
    igpuB <- gpuMatrix(Bint, type="integer")
    
    igpuC <- igpuA + igpuB
    
    expect_is(igpuC, "igpuMatrix")
    expect_equal(igpuC[,], Cint,
                 info="integer matrix elements not equivalent")  
})

test_that("gpuMatrix Double Precision Matrix multiplication successful", {
    
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

test_that("gpuMatrix Double Precision Matrix Subtraction successful", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A - B
    
    dgpuA <- gpuMatrix(A, type="double")
    dgpuB <- gpuMatrix(B, type="double")
    
    dgpuC <- dgpuA - dgpuB
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})

test_that("gpuMatrix Double Precision Matrix Addition successful", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A + B
    
    dgpuA <- gpuMatrix(A, type="double")
    dgpuB <- gpuMatrix(B, type="double")
    
    dgpuC <- dgpuA + dgpuB
    
    expect_is(dgpuC, "dgpuMatrix")
    expect_equal(dgpuC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})
