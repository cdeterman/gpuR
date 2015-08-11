library(gpuR)
context("gpuVector classes")

set.seed(123)

test_that("integer vector class present", {
    
    A <- seq.int(10)    
    gpuA <- gpuVector(A)
    
    expect_is(gpuA, "igpuVector")
    expect_is(gpuA@address, "externalptr")
    expect_that(typeof(gpuA), matches("integer"))
    expect_equivalent(gpuA[,], A)
})

test_that("float vector class present", {
    
    A <- as.numeric(seq(10))
    gpuA <- gpuVector(A, type = "float")
    
    expect_is(gpuA, "fgpuVector")
    expect_is(gpuA@address, "externalptr")
    expect_that(typeof(gpuA), matches("float"))
    expect_equal(gpuA[,], A, tolerance = 1e-07)
})

test_that("double vector class present", {
    
    A <- as.numeric(seq(10))
    gpuA <- gpuVector(A)
    
    expect_is(gpuA, "dgpuVector")
    expect_is(gpuA@address, "externalptr")
    expect_that(typeof(gpuA), matches("double"))
    expect_equal(gpuA[,], A, tolerance = .Machine$double.eps ^ 0.5)
})
