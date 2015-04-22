library(gpuR)
context("Matrix classes")

set.seed(123)
A <- matrix(seq.int(10000), 100)
# B <- sample(seq.int(10000), 100, replace = TRUE)

test_that("integer matrix class present", {
    B <- as.numeric(rnorm(10))
    gpuA <- gpuMatrix(A)
    
    expect_is(gpuA, "igpuMatrix")
    expect_error(gpuMatrix(B))
})
