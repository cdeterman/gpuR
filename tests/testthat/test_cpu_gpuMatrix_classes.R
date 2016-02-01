library(gpuR)
context("CPU gpuMatrix classes")

# set option to use CPU instead of GPU
options(gpuR.default.device.type = "cpu")

set.seed(123)
A <- matrix(seq.int(10000), 100)
D <- matrix(rnorm(100), 10)


test_that("CPU gpuMatrix class contains correct information", {
    
    has_cpu_skip()
    
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

test_that("CPU gpuMatrix integer vector initializers", {
    
    has_cpu_skip()
    
    vi <- seq.int(10)
    v <- rnorm(10)
    Ai <- matrix(vi, nrow=2)
    A <- matrix(v, nrow=5)
    err <- c(TRUE, FALSE)
    err2 <- c("hello", FALSE, 6)
    
    vclAi <- gpuMatrix(vi, nrow=2, ncol=5)
    
    expect_is(vclAi, "igpuMatrix")
    expect_equivalent(vclAi[], Ai)
    expect_equal(dim(Ai), dim(vclAi))
    expect_error(gpuMatrix(err, nrow=1, ncol=2, type="double"))
    expect_error(gpuMatrix(err, nrow=1, ncol=2))
    expect_error(gpuMatrix(err2, nrow=1, ncol=3, type="double"))
    expect_error(gpuMatrix(v, nrow=5, ncol=2, type="integer"))
})

test_that("CPU gpuMatrix float vector initializers", {
    
    has_cpu_skip()
    
    v <- rnorm(10)
    vi <- seq.int(10)
    A <- matrix(v, nrow=5)
    Ai <- matrix(vi, nrow=2)
    
    vclA <- gpuMatrix(v, nrow=5, ncol=2, type="float")
    vclAi <- gpuMatrix(vi, nrow=2, ncol=5, type="float")
    
    expect_equal(vclA[], A, tolerance=1e-07)
    expect_equal(dim(A), dim(vclA))
    expect_equal(vclAi[], Ai, tolerance=1e-07)
    expect_equal(dim(Ai), dim(vclAi))
    expect_is(vclA, "fgpuMatrix")
    expect_is(vclAi, "fgpuMatrix")
})

test_that("CPU gpuMatrix double vector initializers", {
    
    has_cpu_skip()
    
    v <- rnorm(10)
    vi <- seq.int(10)
    A <- matrix(v, nrow=5)
    Ai <- matrix(vi, nrow=2)
    
    vclA <- gpuMatrix(v, nrow=5, ncol=2, type="double")
    vclAi <- gpuMatrix(vi, nrow=2, ncol=5, type="double")
    
    expect_equal(vclA[], A, tolerance=.Machine$double.eps^0.5)
    expect_equal(dim(A), dim(vclA))
    expect_equal(vclAi[], Ai, tolerance=.Machine$double.eps^0.5)
    expect_equal(dim(Ai), dim(vclAi))
    expect_is(vclA, "dgpuMatrix")
    expect_is(vclAi, "dgpuMatrix")
})

test_that("CPU gpuMatrix integer scalar initializers", {
    
    has_cpu_skip()
    
    v <- 3
    vi <- 4L
    A <- matrix(v, nrow=5, ncol=5)
    Ai <- matrix(vi, nrow=2, ncol=7)
    
    ivclA <- gpuMatrix(vi, nrow=2, ncol=7, type="integer")
    
    expect_is(ivclA, "igpuMatrix")
    expect_equivalent(ivclA[], Ai,
                      info = "scalar integer elements not equivalent")
    expect_equivalent(dim(Ai), dim(ivclA),
                      info = "scalar integer dimensions not equivalent")
    expect_error(gpuMatrix(v, nrow=5, ncol=5, type="integer"))
})

test_that("CPU gpuMatrix float scalar initializers", {
    
    has_cpu_skip()
    
    v <- 3
    vi <- 4L
    A <- matrix(v, nrow=5, ncol=5)
    Ai <- matrix(vi, nrow=2, ncol=7)
    
    vclA <- gpuMatrix(v, nrow=5, ncol=5, type="float")
    ivclA <- gpuMatrix(vi, nrow=2, ncol=7, type="float")
    
    expect_equal(vclA[], A, tolerance=1e-07,
                 info="scalar float elements not equivalent")
    expect_equivalent(dim(A), dim(vclA),
                      info="scalar float dimensions not equivalent")
    
    expect_equal(ivclA[], Ai, tolerance=1e-07,
                 info = "integer scalar float elements not equivalent")
    expect_equivalent(dim(Ai), dim(ivclA),
                      info = "integer scalar float dimensions not equivalent")
    expect_is(vclA, "fgpuMatrix")
    expect_is(ivclA, "fgpuMatrix")
})

test_that("CPU gpuMatrix double scalar initializers", {
    
    has_cpu_skip()
    
    v <- 3
    vi <- 4L
    A <- matrix(v, nrow=5, ncol=5)
    Ai <- matrix(vi, nrow=2, ncol=7)
    
    vclA <- gpuMatrix(v, nrow=5, ncol=5, type="double")
    ivclA <- gpuMatrix(vi, nrow=2, ncol=7, type="double")
    
    expect_equal(vclA[], A, tolerance=.Machine$double.eps^0.5,
                 info = "scalar double elements not equivalent")
    expect_equivalent(dim(A), dim(vclA),
                      info = "scalar double dimensions not equivalent")
    expect_equal(ivclA[], Ai, tolerance=.Machine$double.eps^0.5,
                 info = "integer scalar double elements not equivalent")
    expect_equivalent(dim(Ai), dim(ivclA),
                      info = "integer scalar double dimensions not equivalent")
    expect_is(vclA, "dgpuMatrix")
    expect_is(ivclA, "dgpuMatrix")
})


options(gpuR.default.device.type = "gpu")
