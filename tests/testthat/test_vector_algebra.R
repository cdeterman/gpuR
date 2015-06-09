library(gpuR)
context("Vector algebra")

set.seed(123)
ORDER <- 100
A <- sample(seq.int(10), ORDER, replace = TRUE)
B <- sample(seq.int(10), ORDER, replace = TRUE)

test_that("vector additonal successful", {
    
    has_gpu_skip()
    
    gpuA <- as.gpuVector(A)
    gpuB <- as.gpuVector(B)
    
    # R default
    C <- A + B
    
    # manual call
    #gpuC <- gpu_vec_add(A, B)
    
    # generic call
    gpuC <- gpuA + gpuB

    expect_equivalent(gpuC@object, C)
    expect_is(gpuC, "gpuVector", "following vector addition")
    expect_is(gpuC, "igpuVector", "following vector addition")
})

test_that("vector subtraction successful", {
    
    has_gpu_skip()
    
    gpuA <- as.gpuVector(A)
    gpuB <- as.gpuVector(B)
    
    # R default
    C <- A - B
    
    # generic call
    gpuC <- gpuA - gpuB
    
    expect_equivalent(gpuC@object, C)
    expect_is(gpuC, "gpuVector", "following vector subtraction")
    expect_is(gpuC, "igpuVector", "following vector subtraction")
})
