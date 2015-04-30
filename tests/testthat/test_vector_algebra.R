library(gpuR)
context("Vector algebra")

set.seed(123)
A <- sample(seq.int(10), 1000, replace = TRUE)
B <- sample(seq.int(10), 1000, replace = TRUE)

check_for_gpu <- function() {
    if (detectGPUs() == 0) {
        skip("No GPUs available")
    }
}

test_that("vector additonal successful", {
    
    check_for_gpu()
    
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
    
    check_for_gpu()
    
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
