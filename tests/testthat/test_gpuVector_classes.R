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
    expect_equivalent(gpuA[,], A)
    expect_equal(length(gpuA), length(A))
})

test_that("float vector class present", {
    
    has_gpu_skip()
    
    A <- as.numeric(seq(10))
    gpuA <- gpuVector(A, type = "float")
    
    expect_is(gpuA, "fgpuVector")
    expect_is(gpuA@address, "externalptr")
    expect_that(typeof(gpuA), matches("float"))
    expect_equal(gpuA[,], A, tolerance = 1e-07)
    expect_equal(length(gpuA), length(A))
})

test_that("double vector class present", {
    
    has_gpu_skip()
    has_double_skip()
    
    A <- as.numeric(seq(10))
    gpuA <- gpuVector(A)
    
    expect_is(gpuA, "dgpuVector")
    expect_is(gpuA@address, "externalptr")
    expect_that(typeof(gpuA), matches("double"))
    expect_equal(gpuA[,], A, tolerance = .Machine$double.eps ^ 0.5)
    expect_equal(length(gpuA), length(A))
})

test_that("fgpuVectorSlice class present", {
    has_gpu_skip()
    
    A <- as.numeric(seq(10))
    S <- A[2:8]
    gpuA <- gpuVector(A, type = "float")
    gpuS <- slice(gpuA, 2L, 8L)
    
    expect_is(gpuS, "gpuVector")
    expect_is(gpuS, "fgpuVectorSlice")
    expect_is(gpuS@address, "externalptr")
    expect_that(typeof(gpuS), matches("float"))
    expect_equal(gpuS[,], S, tolerance = 1e-07)
    expect_equal(length(gpuS), length(S))
    
    
    # check that slice refers back to original vector
    gpuS[3] <- 42.42
    S[3] <- 42.42
    
    expect_equal(gpuS[], S, tolerance = 1e-07)
    expect_false(isTRUE(all.equal(gpuA[], A, tolerance = 1e-07)),
                 info = "source fgpuVector not modified by slice")
    expect_equal(length(gpuA), length(A), 
                 info = "source fgpuVector length has been changed")
})

test_that("dgpuVectorSlice class present", {
    has_gpu_skip()
    has_double_skip()
    
    A <- as.numeric(seq(10))
    S <- A[2:8]
    gpuA <- gpuVector(A, type = "double")
    gpuS <- slice(gpuA, 2L, 8L)
    
    expect_is(gpuS, "gpuVector")
    expect_is(gpuS, "dgpuVectorSlice")
    expect_is(gpuS@address, "externalptr")
    expect_that(typeof(gpuS), matches("double"))
    expect_equal(gpuS[,], S, tolerance = .Machine$double.eps^0.5)
    expect_equal(length(gpuS), length(S))
    
    
    # check that slice refers back to original vector
    gpuS[3] <- 42.42
    S[3] <- 42.42
    
    expect_equal(gpuS[], S, tolerance = .Machine$double.eps^0.5)
    expect_false(isTRUE(all.equal(gpuA[], A, tolerance = .Machine$double.eps^0.5)),
                 info = "source dgpuVector not modified by slice")
    expect_equal(length(gpuA), length(A), 
                 info = "source dgpuVector length has been changed")
})

