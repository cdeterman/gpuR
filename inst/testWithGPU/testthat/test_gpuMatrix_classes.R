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

test_that("gpuMatrix integer vector initializers", {
    
    has_gpu_skip()
    
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

test_that("gpuMatrix float vector initializers", {
    
    has_gpu_skip()
    
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

test_that("gpuMatrix double vector initializers", {
    
    has_gpu_skip()
    has_double_skip()
    
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

test_that("gpuMatrix integer scalar initializers", {
    
    has_gpu_skip()
    
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

test_that("gpuMatrix float scalar initializers", {
    
    has_gpu_skip()
    
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

test_that("gpuMatrix double scalar initializers", {
    
    has_gpu_skip()
    has_double_skip()
    
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

test_that("fgpuMatrixBlock class present", {
    
    has_gpu_skip()
    
    S <- A[2:8, 2:10]
    gpuA <- gpuMatrix(A, type = "float")
    gpuS <- block(gpuA, 2L, 8L, 2L, 10L)
    
    expect_is(gpuS, "gpuMatrix")
    expect_is(gpuS, "fgpuMatrixBlock")
    expect_is(gpuS@address, "externalptr")
    expect_that(typeof(gpuS), matches("float"))
    expect_equal(gpuS[,], S, tolerance = 1e-07)
    expect_equal(dim(gpuS), dim(S))
    
    
    # check that block refers back to original vector
    gpuS[3,3] <- 42.42
    S[3,3] <- 42.42
    
    expect_equal(gpuS[], S, tolerance = 1e-07)
    expect_false(isTRUE(all.equal(gpuA[], A, tolerance = 1e-07)),
                 info = "source fgpuMatrix not modified by block")
    expect_equal(dim(gpuA), dim(A), 
                 info = "source fgpuMatrix dimensions have been changed")
})

test_that("dgpuMatrixBlock class present", {
    has_gpu_skip()
    has_double_skip()
    
    S <- A[2:8, 2:10]
    gpuA <- gpuMatrix(A, type = "double")
    gpuS <- block(gpuA, 2L, 8L, 2L, 10L)
    
    expect_is(gpuS, "gpuMatrix")
    expect_is(gpuS, "dgpuMatrixBlock")
    expect_is(gpuS@address, "externalptr")
    expect_that(typeof(gpuS), matches("double"))
    expect_equal(gpuS[,], S, tolerance = .Machine$double.eps^0.5)
    expect_equal(dim(gpuS), dim(S))
    
    
    # check that block refers back to original vector
    gpuS[3,3] <- 42.42
    S[3,3] <- 42.42
    
    expect_equal(gpuS[], S, tolerance = .Machine$double.eps^0.5)
    expect_false(isTRUE(all.equal(gpuA[], A, tolerance = .Machine$double.eps^0.5)),
                 info = "source dgpuMatrix not modified by block")
    expect_equal(dim(gpuA), dim(A), 
                 info = "source dgpuMatrix dimensions been changed")
})
