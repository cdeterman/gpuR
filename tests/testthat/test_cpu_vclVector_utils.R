library(gpuR)
context("CPU vclVector Utility Functions")

# set option to use CPU instead of GPU
options(gpuR.default.device = "cpu")

set.seed(123)
ORDER <- 100
A <- sample(seq.int(10), ORDER, replace = TRUE)
D <- rnorm(ORDER)

test_that("integer vclVector length method successful", {
    
    gpuA <- vclVector(A)
    
    s <- length(gpuA)
    
    expect_true(s == ORDER)
})

test_that("float vclVector length method successful", {
    
    gpuA <- vclVector(A, type="float")
    
    s <- length(gpuA)
    
    expect_true(s == ORDER)
})

test_that("double vclVector length method successful", {
    
    gpuA <- vclVector(A, type="double")
    
    s <- length(gpuA)
    
    expect_true(s == ORDER)
})

test_that("vclVector accession method successful", {
    
    gpuA <- vclVector(A)
    gpuF <- vclVector(D, type="float")
    
    gi <- gpuA[2]
    i <- A[2]
    gs <- gpuF[2]
    s <- D[2]
    
    expect_equivalent(gi, i, info = "ivclVector element access not correct")
    expect_equal(gs, s, tolerance = 1e-07, 
                 info = "fvclVector element access not correct")
    expect_error(gpuA[101], info = "no error when outside vclVector size")
})

test_that("dvclVector accession method successful", {
    
    gpuD <- vclVector(D)
    
    gs <- gpuD[2]
    s <- D[2]
    
    expect_equivalent(gs, s, info = "dvclVector element access not correct")
})

options(gpuR.default.device = "gpu")
