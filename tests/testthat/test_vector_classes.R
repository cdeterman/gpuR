library(gpuR)
context("Vector classes")

set.seed(123)
A <- sample(seq.int(10), 1000, replace = TRUE)
B <- sample(seq.int(10), 1000, replace = TRUE)

test_that("integer vector class present", {
    A <- seq.int(10)    
    B <- as.numeric(seq(10))
    gpuA <- gpuVector(A)
    
    expect_is(gpuA, "igpuVector")
    expect_error(gpuVector(B))
})
