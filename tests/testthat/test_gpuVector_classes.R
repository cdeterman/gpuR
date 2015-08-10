library(gpuR)
context("gpuVector classes")

set.seed(123)

test_that("integer vector class present", {
    
    has_gpu_skip()
    
    A <- seq.int(10)    
    gpuA <- gpuVector(A)
    
    expect_is(gpuA, "igpuVector")
    expect_is(gpuA@address, "externalptr")
    expect_that(typeof(gpuA), matches("integer"))
})

test_that("float vector class present", {
    
    has_gpu_skip()
    
    A <- as.numeric(seq(10))
    gpuA <- gpuVector(A, type = "float")
    
    expect_is(gpuA, "fgpuVector")
    expect_is(gpuA@address, "externalptr")
    expect_that(typeof(gpuA), matches("float"))
})

test_that("double vector class present", {
    
    has_gpu_skip()
    
    A <- as.numeric(seq(10))
    gpuA <- gpuVector(A)
    
    expect_is(gpuA, "dgpuVector")
    expect_is(gpuA@address, "externalptr")
    expect_that(typeof(gpuA), matches("double"))
})
