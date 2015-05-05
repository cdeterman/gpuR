library(gpuR)
context("gpuBigMatrix algebra")

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

Cint <- Aint %*% Bint
C <- A %*% B

test_that("gpuBigMatrix Integer Matrix Multiplication successful", {
    
    has_gpu_skip()
    
    igpuA <- gpuBigMatrix(Aint, type="integer")
    igpuB <- gpuBigMatrix(Bint, type="integer")
    
    igpuC <- igpuA %*% igpuB
    
    expect_is(igpuC, "igpuBigMatrix")
    expect_equivalent(dim(igpuC), dim(Cint), 
                      "integer matrix dimensions not equivalent")
    expect_equal(igpuC[,], Cint, 
                 info="integer matrix elements not equivalent")
})

test_that("gpuBigMatrix Single Precision Matrix multiplication successful", {
    
    has_gpu_skip()
    
    # GPU matrix objects
    fgpuA <- gpuBigMatrix(A, type="float")
    fgpuB <- gpuBigMatrix(B, type="float")
    
    fgpuC <- fgpuA %*% fgpuB

    expect_is(fgpuC, "fgpuBigMatrix")
    expect_equivalent(dim(fgpuC), dim(C), 
                      "float matrix dimensions not equivalent")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")
})

test_that("gpuBigMatrix Single Precision Big Matrix Addition successful", {
    
    has_gpu_skip()
    
    C <- A + B
    
    fgpuA <- gpuBigMatrix(A, type="float")
    fgpuB <- gpuBigMatrix(B, type="float")
    
    fgpuC <- fgpuA + fgpuB
    
    expect_is(fgpuC, "fgpuBigMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("gpuBigMatrix Single Precision Big Matrix Subtraction successful", {
    
    has_gpu_skip()
    
    C <- A - B
    
    fgpuA <- gpuBigMatrix(A, type="float")
    fgpuB <- gpuBigMatrix(B, type="float")
    
    fgpuC <- fgpuA - fgpuB
    
    expect_is(fgpuC, "fgpuBigMatrix")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("gpuBigMatrix Integer Big Matrix Addition successful", {
    
    has_gpu_skip()
    
    Cint <- Aint + Bint
    
    igpuA <- gpuBigMatrix(Aint, type="integer")
    igpuB <- gpuBigMatrix(Bint, type="integer")
    
    igpuC <- igpuA + igpuB
    
    expect_is(igpuC, "igpuBigMatrix")
    expect_equal(igpuC[,], Cint,
                 info="integer matrix elements not equivalent")  
})

test_that("gpuBigMatrix Integer Big Matrix Subtraction successful", {
    
    has_gpu_skip()
    
    Cint <- Aint - Bint
    
    igpuA <- gpuBigMatrix(Aint, type="integer")
    igpuB <- gpuBigMatrix(Bint, type="integer")
    
    igpuC <- igpuA - igpuB
    
    expect_is(igpuC, "igpuBigMatrix")
    expect_equal(igpuC[,], Cint,
                 info="integer matrix elements not equivalent")  
})
