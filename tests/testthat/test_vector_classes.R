library(bigGPU)
context("vector classes")

test_that("integer vector class present", {
    A <- seq.int(10)    
    B <- as.numeric(seq(10))
    gpuA <- gpuVector(A)
    
    expect_is(gpuA, "igpuVector")
    expect_error(gpuVector(B))
})
