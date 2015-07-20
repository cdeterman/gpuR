library(gpuR)
context("Vector algebra")

set.seed(123)
ORDER <- 100
A <- sample(seq.int(10), ORDER, replace = TRUE)
B <- sample(seq.int(10), ORDER, replace = TRUE)

test_that("integer vector additonal successful", {
    
    has_gpu_skip()
    
    gpuA <- gpuVector(A)
    gpuB <- gpuVector(B)
    
    # R default
    C <- A + B
    
    # manual call
    #gpuC <- gpu_vec_add(A, B)
    
    # generic call
    gpuC <- gpuA + gpuB

    expect_equivalent(gpuC[], C)
    expect_is(gpuC, "gpuVector", "inherits from gpuVector")
    expect_is(gpuC, "igpuVector", "is a igpuVector object")
})

test_that("integer vector subtraction successful", {
    
    has_gpu_skip()
    
    gpuA <- gpuVector(A)
    gpuB <- gpuVector(B)
    
    # R default
    C <- A - B
    
    # generic call
    gpuC <- gpuA - gpuB
    
    expect_equivalent(gpuC[], C)
    expect_is(gpuC, "gpuVector", "following vector subtraction")
    expect_is(gpuC, "igpuVector", "following vector subtraction")
})
