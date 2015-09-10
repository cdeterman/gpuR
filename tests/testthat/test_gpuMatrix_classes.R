library(gpuR)
context("gpuMatrix classes")

set.seed(123)
A <- matrix(seq.int(10000), 100)
D <- matrix(rnorm(100), 10)


test_that("gpuMatrix class contains correct information", {
    
    has_gpu_skip()
    has_double_skip()
    
    B <- as.numeric(rnorm(10))
    gpuA <- gpuMatrix(A)
    
    expect_is(gpuA, "igpuMatrix")
    expect_true(typeof(gpuA) == "integer")
    expect_equivalent(gpuA[,], A)
    
    gpuD <- gpuMatrix(D)
    expect_is(gpuD, "dgpuMatrix")
    expect_true(typeof(gpuD) == "double")
    expect_equivalent(gpuD[,], D)
    
    gpuF <- gpuMatrix(D, type="float")
    expect_is(gpuF, "fgpuMatrix")
    expect_true(typeof(gpuF) == "float")
    expect_equal(gpuF[,], D, tolerance = 1e-07)
    
    # can't convert a vector to a gpuMatrix
    expect_error(gpuMatrix(B))
})

test_that("gpuMatrix vector initializers", {
    
    has_gpu_skip()
    has_double_skip()
    
    v <- rnorm(10)
    vi <- seq.int(10)
    A <- matrix(v, nrow=5)
    Ai <- matrix(vi, nrow=2)
    err <- c(TRUE, FALSE)
    err2 <- c("hello", FALSE, 6)
    
    gpuA <- gpuMatrix(v, nrow=5, ncol=2, type="double")
    gpuAi <- gpuMatrix(vi, nrow=2, ncol=5)
    
    expect_equivalent(gpuA[], A)
    expect_equal(dim(A), dim(gpuA))
    expect_is(gpuAi, "igpuMatrix")
    expect_equivalent(gpuAi[], Ai)
    expect_equal(dim(Ai), dim(gpuAi))
    expect_error(gpuMatrix(err, nrow=1, ncol=2, type="double"))
    expect_error(gpuMatrix(err, nrow=1, ncol=2))
    expect_error(gpuMatrix(err2, nrow=1, ncol=3, type="double"))
})

