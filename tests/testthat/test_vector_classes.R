library(gpuR)
context("vector classes")

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

test_that("gpuVector class returned from Arith methods", {
    gpuA <- as.gpuVector(A)
    gpuB <- as.gpuVector(B)
    
    # generic call
    gpuC <- gpuA + gpuB
    gpuC2 <- gpuA - gpuB
    expect_is(gpuC, "gpuVector")
    expect_is(gpuC, "igpuVector")
    expect_is(gpuC2, "igpuVector")
})
