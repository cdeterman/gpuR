library(gpuR)
context("Switching GPUs vclMatrix classes")

set.seed(123)
A <- matrix(seq.int(100), nrow=5)
D <- matrix(rnorm(100), nrow=5)
v <- rnorm(100)
vi <- seq.int(100)


test_that("Switching GPUs vclMatrix integer class initializer" ,{
    
    has_multiple_gpu_skip()
    
    setContext(2L)
    
    vclA <- vclMatrix(A)
    
    setContext(1L)
    
    expect_is(vclA, "ivclMatrix")
    expect_equivalent(vclA[], A, 
                      info="vcl integer matrix elements not equivalent")
    expect_equal(dim(vclA), dim(A))
    expect_equal(ncol(vclA), ncol(A))
    expect_equal(nrow(vclA), nrow(A))
    expect_equal(typeof(vclA), "integer")
    expect_equal(vclA@.context_index, 2L,
                 info = "context index not set correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix float class initializer" ,{
    
    has_multiple_gpu_skip()
    
    setContext(2L)
    
    vclD <- vclMatrix(D, type="float")
    
    setContext(1L)
    
    expect_is(vclD, "fvclMatrix")
    expect_equal(vclD[], D, tolerance=1e-07, 
                 info="vcl float matrix elements not equivalent")
    expect_equal(dim(vclD), dim(D))
    expect_equal(ncol(vclD), ncol(D))
    expect_equal(nrow(vclD), nrow(D))
    expect_equal(typeof(vclD), "float")
    expect_equal(vclD@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix double class initializer" ,{
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    setContext(2L)
    
    vclD <- vclMatrix(D)
    
    setContext(1L)
    
    expect_is(vclD, "dvclMatrix")
    expect_equal(vclD[], D, tolerance=.Machine$double.eps ^ 0.5, 
                 info="vcl double matrix elements not equivalent")
    expect_equal(dim(vclD), dim(D))
    expect_equal(ncol(vclD), ncol(D))
    expect_equal(nrow(vclD), nrow(D))
    expect_equal(typeof(vclD), "double")    
    expect_equal(vclD@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix integer vector initializers", {
    
    has_multiple_gpu_skip()
    
    vi <- seq.int(10)
    Ai <- matrix(vi, nrow=2)
    
    setContext(2L)
    
    vclAi <- vclMatrix(vi, nrow=2, ncol=5)
    
    setContext(1L)
    
    expect_is(vclAi, "ivclMatrix")
    expect_equivalent(vclAi[], Ai)
    expect_equal(dim(Ai), dim(vclAi))
    expect_equal(vclAi@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix float vector initializers", {
    
    has_multiple_gpu_skip()
    
    v <- rnorm(10)
    A <- matrix(v, nrow=5)
    
    setContext(2L)
    
    vclA <- vclMatrix(v, nrow=5, ncol=2, type="float")
    
    setContext(1L)
    
    expect_equal(vclA[], A, tolerance=1e-07)
    expect_equal(dim(A), dim(vclA))
    expect_is(vclA, "fvclMatrix")
    expect_equal(vclA@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix double vector initializers", {
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    v <- rnorm(10)
    A <- matrix(v, nrow=5)
    
    setContext(2L)
    
    vclA <- vclMatrix(v, nrow=5, ncol=2, type="double")
    
    setContext(1L)
    
    expect_equal(vclA[], A, tolerance=.Machine$double.eps^0.5)
    expect_equal(dim(A), dim(vclA))
    expect_is(vclA, "dvclMatrix")
    expect_equal(vclA@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix integer scalar initializers", {
    
    has_multiple_gpu_skip()
    
    vi <- 4L
    Ai <- matrix(vi, nrow=2, ncol=7)
    
    setContext(2L)
    
    ivclA <- vclMatrix(vi, nrow=2, ncol=7, type="integer")
    
    setContext(1L)
    
    expect_error(vclMatrix(v, nrow=5, ncol=5, type="integer"))
    expect_is(ivclA, "ivclMatrix")
    expect_equivalent(ivclA[], Ai,
                      "scalar integer elements not equivalent")
    expect_equivalent(dim(Ai), dim(ivclA),
                      "scalar integer dimensions not equivalent")
    expect_equal(ivclA@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix float scalar initializers", {
    
    has_multiple_gpu_skip()
    
    v <- 3
    A <- matrix(v, nrow=5, ncol=5)
    
    setContext(2L)
    
    vclA <- vclMatrix(v, nrow=5, ncol=5, type="float")
    
    setContext(1L)
    
    expect_is(vclA, "fvclMatrix")
    expect_equal(vclA[], A, tolerance=1e-07,
                 info="scalar double elements not equivalent")
    expect_equivalent(dim(A), dim(vclA),
                      info="scalar double dimensions not equivalent")
    expect_equal(vclA@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix double scalar initializers", {
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    v <- 3
    A <- matrix(v, nrow=5, ncol=5)
    
    setContext(2L)
    
    vclA <- vclMatrix(v, nrow=5, ncol=5, type="double")
    
    setContext(1L)
    
    expect_is(vclA, "dvclMatrix")
    expect_equal(vclA[], A, tolerance=.Machine$double.eps^0.5,
                 info = "scalar double elements not equivalent")
    expect_equivalent(dim(A), dim(vclA),
                      info = "scalar double dimensions not equivalent")
    expect_equal(vclA@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})
