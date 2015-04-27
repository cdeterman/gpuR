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

# print("R A matrix")
# print(Aint)
# print("R B matrix")
# print(Bint)

Cint <- Aint %*% Bint

test_that("gpuMatrix Integer Matrix multiplication successful", {
    
    igpuA <- gpuMatrix(Aint, type="integer")
    igpuB <- gpuMatrix(Bint, type="integer")
    
    igpuC <- igpuA %*% igpuB
    
    expect_equivalent(igpuC@x[,], Cint, 
                      info="float matrix elements not equivalent")      
})

test_that("gpuMatrix Single Precision Matrix multiplication successful", {
    
    C <- A %*% B
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    
    fgpuC <- fgpuA %*% fgpuB
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC@x[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("gpuMatrix Single Precision Matrix Subtraction successful", {
    
    C <- A - B
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")

    fgpuC <- fgpuA - fgpuB
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC@x[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("gpuMatrix Single Precision Matrix Addition successful", {
    
    C <- A + B
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    
    fgpuC <- fgpuA + fgpuB
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuC@x[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("gpuMatrix Integer Matrix Subtraction successful", {
    
    Cint <- Aint - Bint
    
    igpuA <- gpuMatrix(Aint, type="integer")
    igpuB <- gpuMatrix(Bint, type="integer")
    
    igpuC <- igpuA - igpuB
    
    expect_is(igpuC, "igpuMatrix")
    expect_equal(igpuC@x[,], Cint, 
                 info="integer matrix elements not equivalent")  
})

test_that("gpuMatrix Integer Matrix Addition successful", {
    
    Cint <- Aint + Bint
    
    igpuA <- gpuMatrix(Aint, type="integer")
    igpuB <- gpuMatrix(Bint, type="integer")
    
    igpuC <- igpuA + igpuB
    
    expect_is(igpuC, "igpuMatrix")
    expect_equal(igpuC@x[,], Cint,
                 info="integer matrix elements not equivalent")  
})
