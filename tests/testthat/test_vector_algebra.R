library(gpuR)
context("vector algebra")

set.seed(123)
A <- sample(seq.int(10), 1000, replace = TRUE)
B <- sample(seq.int(10), 1000, replace = TRUE)


# test_that("can read cl file", {
#     print(test())
# })

test_that("vector additonal successful", {
    gpuA <- as.gpuVector(A)
    gpuB <- as.gpuVector(B)
    
    # R default
    C <- A + B
    
    # manual call
    #gpuC <- gpu_vec_add(A, B)
    
    # generic call
    gpuC <- gpuA + gpuB

    expect_equivalent(gpuC@object, C)
})

test_that("vector subtraction successful", {
    gpuA <- as.gpuVector(A)
    gpuB <- as.gpuVector(B)
    
    # R default
    C <- A - B
    
    # generic call
    gpuC <- gpuA - gpuB
    
    expect_equivalent(gpuC@object, C)
})
