library(gpuR)
context("gpuVector Utility Functions")

set.seed(123)
ORDER <- 100
A <- sample(seq.int(10), ORDER, replace = TRUE)

test_that("integer vector length method successful", {
    
    has_gpu_skip()
    
    gpuA <- gpuVector(A)
    
    s <- length(gpuA)
    
    expect_true(s == ORDER)
})

test_that("float vector length method successful", {
    
    has_gpu_skip()
    
    gpuA <- gpuVector(A, type="float")
    
    s <- length(gpuA)
    
    expect_true(s == ORDER)
})

test_that("double vector length method successful", {
    
    has_gpu_skip()
    has_double_skip()
    
    gpuA <- gpuVector(A, type="double")
    
    s <- length(gpuA)
    
    expect_true(s == ORDER)
})
