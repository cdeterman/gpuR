library(bigGPU)
context("vector classes")

A <- seq.int(from=0, to=999)
B <- seq.int(from=1000, to=1)

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
    expect_is(gpuC, "gpuVector")
    expect_is(gpuC, "igpuVector")
})
