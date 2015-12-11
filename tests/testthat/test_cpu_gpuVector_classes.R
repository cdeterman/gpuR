library(gpuR)
context("CPU gpuVector classes")

# set option to use CPU instead of GPU
options(gpuR.default.device = "cpu")

set.seed(123)

test_that("integer vector class present", {
    has_cpu_skip()
    
    A <- seq.int(10)    
    gpuA <- gpuVector(A)
    
    expect_is(gpuA, "igpuVector")
    expect_is(gpuA@address, "externalptr")
    expect_that(typeof(gpuA), matches("integer"))
    expect_equivalent(gpuA[,], A)
    expect_equal(length(gpuA), length(A))
})

test_that("float vector class present", {
    has_cpu_skip()
    
    A <- as.numeric(seq(10))
    gpuA <- gpuVector(A, type = "float")
    
    expect_is(gpuA, "fgpuVector")
    expect_is(gpuA@address, "externalptr")
    expect_that(typeof(gpuA), matches("float"))
    expect_equal(gpuA[,], A, tolerance = 1e-07)
    expect_equal(length(gpuA), length(A))
})

test_that("double vector class present", {
    has_cpu_skip()
    
    A <- as.numeric(seq(10))
    gpuA <- gpuVector(A)
    
    expect_is(gpuA, "dgpuVector")
    expect_is(gpuA@address, "externalptr")
    expect_that(typeof(gpuA), matches("double"))
    expect_equal(gpuA[,], A, tolerance = .Machine$double.eps ^ 0.5)
    expect_equal(length(gpuA), length(A))
})

test_that("fgpuVectorSlice class present", {
    has_cpu_skip()
    
    A <- as.numeric(seq(10))
    gpuA <- gpuVector(A, type = "float")
    gpuS <- slice(gpuA, 2L, 8L)
    
    expect_is(gpuS, "gpuVector")
    expect_is(gpuS, "gpuVectorSlice")
    expect_is(gpuS, "fgpuVectorSlice")
    expect_is(gpuS@address, "externalptr")
    expect_that(typeof(gpuS), matches("float"))
    expect_equal(gpuS[,], A[2:8], tolerance = 1e-07)
    expect_equal(length(gpuS), length(A[2:8]))
})

options(gpuR.default.device = "gpu")
