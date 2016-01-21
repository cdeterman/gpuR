library(gpuR)
context("vclMatrix classes")

set.seed(123)
A <- matrix(seq.int(100), nrow=5)
D <- matrix(rnorm(100), nrow=5)
v <- rnorm(100)
vi <- seq.int(100)


test_that("vclMatrix integer class initializer" ,{
    
    has_gpu_skip()
    
    vclA <- vclMatrix(A)
    
    expect_is(vclA, "ivclMatrix")
    expect_equivalent(vclA[], A, 
                      info="vcl integer matrix elements not equivalent")
    expect_equal(dim(vclA), dim(A))
    expect_equal(ncol(vclA), ncol(A))
    expect_equal(nrow(vclA), nrow(A))
    expect_equal(typeof(vclA), "integer")
})

test_that("vclMatrix float class initializer" ,{
    
    has_gpu_skip()
    
    vclD <- vclMatrix(D, type="float")
    
    expect_is(vclD, "fvclMatrix")
    expect_equal(vclD[], D, tolerance=1e-07, 
                      info="vcl float matrix elements not equivalent")
    expect_equal(dim(vclD), dim(D))
    expect_equal(ncol(vclD), ncol(D))
    expect_equal(nrow(vclD), nrow(D))
    expect_equal(typeof(vclD), "float")
})

test_that("vclMatrix double class initializer" ,{
    
    has_gpu_skip()
    has_double_skip()
    
    vclD <- vclMatrix(D)
    
    expect_is(vclD, "dvclMatrix")
    expect_equal(vclD[], D, tolerance=.Machine$double.eps ^ 0.5, 
                      info="vcl double matrix elements not equivalent")
    expect_equal(dim(vclD), dim(D))
    expect_equal(ncol(vclD), ncol(D))
    expect_equal(nrow(vclD), nrow(D))
    expect_equal(typeof(vclD), "double")
})

test_that("vclMatrix integer vector initializers", {
    
    has_gpu_skip()
    
    vi <- seq.int(10)
    Ai <- matrix(vi, nrow=2)
    err <- c(TRUE, FALSE)
    err2 <- c("hello", FALSE, 6)
    
    vclAi <- vclMatrix(vi, nrow=2, ncol=5)
    
    expect_is(vclAi, "ivclMatrix")
    expect_equivalent(vclAi[], Ai)
    expect_equal(dim(Ai), dim(vclAi))
    expect_error(vclMatrix(err, nrow=1, ncol=2, type="double"))
    expect_error(vclMatrix(err, nrow=1, ncol=2))
    expect_error(vclMatrix(err2, nrow=1, ncol=3, type="double"))
})

test_that("vclMatrix float vector initializers", {
    
    has_gpu_skip()
    
    v <- rnorm(10)
    A <- matrix(v, nrow=5)
    
    vclA <- vclMatrix(v, nrow=5, ncol=2, type="float")
    
    expect_equal(vclA[], A, tolerance=1e-07)
    expect_equal(dim(A), dim(vclA))
    expect_is(vclA, "fvclMatrix")
})

test_that("vclMatrix double vector initializers", {
    
    has_gpu_skip()
    has_double_skip()
    
    v <- rnorm(10)
    A <- matrix(v, nrow=5)
    
    vclA <- vclMatrix(v, nrow=5, ncol=2, type="double")
    
    expect_equal(vclA[], A, tolerance=.Machine$double.eps^0.5)
    expect_equal(dim(A), dim(vclA))
    expect_is(vclA, "dvclMatrix")
})

test_that("vclMatrix integer scalar initializers", {
    
    has_gpu_skip()
    
    vi <- 4L
    Ai <- matrix(vi, nrow=2, ncol=7)
    
    ivclA <- vclMatrix(vi, nrow=2, ncol=7, type="integer")
    
    expect_error(vclMatrix(v, nrow=5, ncol=5, type="integer"))
    
    expect_is(ivclA, "ivclMatrix")
    expect_equivalent(ivclA[], Ai,
                      "scalar integer elements not equivalent")
    expect_equivalent(dim(Ai), dim(ivclA),
                 "scalar integer dimensions not equivalent")
})

test_that("vclMatrix float scalar initializers", {
    
    has_gpu_skip()
    
    v <- 3
    A <- matrix(v, nrow=5, ncol=5)
    
    vclA <- vclMatrix(v, nrow=5, ncol=5, type="float")
    
    expect_equal(vclA[], A, tolerance=1e-07,
                 info="scalar double elements not equivalent")
    expect_equivalent(dim(A), dim(vclA),
                 info="scalar double dimensions not equivalent")
    expect_is(vclA, "fvclMatrix")
})

test_that("vclMatrix double scalar initializers", {
    
    has_gpu_skip()
    has_double_skip()
    
    v <- 3
    A <- matrix(v, nrow=5, ncol=5)
    
    vclA <- vclMatrix(v, nrow=5, ncol=5, type="double")
    
    expect_equal(vclA[], A, tolerance=.Machine$double.eps^0.5,
                 info = "scalar double elements not equivalent")
    expect_equivalent(dim(A), dim(vclA),
                 info = "scalar double dimensions not equivalent")
    expect_is(vclA, "dvclMatrix")
})

