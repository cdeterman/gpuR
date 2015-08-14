library(gpuR)
context("gpuVector algebra")

set.seed(123)
ORDER <- 4
Aint <- sample(seq.int(10), ORDER, replace = TRUE)
Bint <- sample(seq.int(10), ORDER, replace = TRUE)
A <- rnorm(ORDER)
B <- rnorm(ORDER)


test_that("integer vector additonal successful", {
    
    has_gpu_skip()
    
    gpuA <- gpuVector(Aint)
    gpuB <- gpuVector(Bint)
    
    # R default
    C <- Aint + Bint
    
    # manual call
    #gpuC <- gpu_vec_add(A, B)
    
    # generic call
    gpuC <- gpuA + gpuB

    expect_equivalent(gpuC[], C)
    expect_is(gpuC, "gpuVector", "inherits from gpuVector")
    expect_is(gpuC, "igpuVector", "is a igpuVector object")
})

test_that("integer vector subtraction successful", {
    
    has_gpu_skip()
    
    gpuA <- gpuVector(Aint)
    gpuB <- gpuVector(Bint)
    
    # R default
    C <- Aint - Bint
    
    # generic call
    gpuC <- gpuA - gpuB
    
    expect_equivalent(gpuC[], C)
    expect_is(gpuC, "gpuVector", "following vector subtraction")
    expect_is(gpuC, "igpuVector", "following vector subtraction")
})

test_that("single precision vector additonal successful", {
    
    has_gpu_skip()
    
    gpuA <- gpuVector(A, type="float")
    gpuB <- gpuVector(B, type="float")
    
    # R default
    C <- A + B
    
    # generic call
    gpuC <- gpuA + gpuB
    
    expect_equal(gpuC[], C, tolerance=1e-07)
    expect_is(gpuC, "fgpuVector",
              info="is not a fgpuVector object")
})

test_that("single precision vector subtraction successful", {
    
    has_gpu_skip()
    
    gpuA <- gpuVector(A, type="float")
    gpuB <- gpuVector(B, type="float")
    
    # R default
    C <- A - B
    
    # generic call
    gpuC <- gpuA - gpuB
    
#     print(A)
#     print(B)
#     print(C)
#     print(gpuC[])
    
    expect_equal(gpuC[], C, tolerance=1e-07)
    expect_is(gpuC, "fgpuVector",
              info = "not a fgpuVector object")
})

test_that("double precision vector additonal successful", {
    
    has_gpu_skip()
    has_double_skip()
    
    gpuA <- gpuVector(A, type="double")
    gpuB <- gpuVector(B, type="double")
    
    # R default
    C <- A + B
    
    # generic call
    gpuC <- gpuA + gpuB
    
    expect_equal(gpuC[], C, tolerance=.Machine$double.eps ^ 0.5)
    expect_is(gpuC, "dgpuVector",
              info="is not a dgpuVector object")
})

test_that("double precision vector subtraction successful", {
    
    has_gpu_skip()
    
    gpuA <- gpuVector(A, type="double")
    gpuB <- gpuVector(B, type="double")
    
    # R default
    C <- A - B
    
    # generic call
    gpuC <- gpuA - gpuB
    
    expect_equal(gpuC[], C, tolerance=.Machine$double.eps ^ 0.5)
    expect_is(gpuC, "dgpuVector", 
              info="is not a dgpuVector object")
})


test_that("gpuVector single precision inner product", {
    C <- A %*% B
    gpuA <- gpuVector(A, type="float")
    gpuB <- gpuVector(B, type="float")
    
    gpuC <- gpuA %*% gpuB
    
    expect_is(gpuC, "matrix")
    expect_equal(gpuC, C, tolerance=1e-06, 
                 info="float vector inner product elements not equivalent")
})

test_that("gpuVector double precision inner product", {
    C <- A %*% B
    gpuA <- gpuVector(A, type="double")
    gpuB <- gpuVector(B, type="double")
    
    gpuC <- gpuA %*% gpuB
    
    expect_is(gpuC, "matrix")
    expect_equal(gpuC, C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vector inner product elements not equivalent")
})

test_that("gpuVector single precision outer product", {
    C <- A %o% B
    gpuA <- gpuVector(A, type="float")
    gpuB <- gpuVector(B, type="float")
    
    gpuC <- gpuA %o% gpuB
    
    expect_is(gpuC, "fgpuMatrix")
    expect_equal(gpuC[], C, tolerance=1e-06, 
                 info="float vector outer product elements not equivalent")
})

test_that("gpuVector double precision outer product", {
    C <- A %o% B
    gpuA <- gpuVector(A, type="double")
    gpuB <- gpuVector(B, type="double")
    
    gpuC <- gpuA %o% gpuB
    
    expect_is(gpuC, "dgpuMatrix")
    expect_equal(gpuC[], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vector outer product elements not equivalent")
})

