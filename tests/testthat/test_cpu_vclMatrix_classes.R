library(gpuR)
context("CPU vclMatrix classes")

# set option to use CPU instead of GPU
options(gpuR.default.device = "cpu")

set.seed(123)
A <- matrix(seq.int(100), nrow=5)
D <- matrix(rnorm(100), nrow=5)
v <- rnorm(100)
vi <- seq.int(100)


test_that("CPU vclMatrix integer class initializer" ,{
    
    has_cpu_skip()
    
    vclA <- vclMatrix(A)
    
    expect_is(vclA, "ivclMatrix")
    expect_equivalent(vclA[], A, 
                      info="vcl integer matrix elements not equivalent")
    expect_equal(dim(vclA), dim(A))
    expect_equal(ncol(vclA), ncol(A))
    expect_equal(nrow(vclA), nrow(A))
    expect_equal(typeof(vclA), "integer")
})

test_that("CPU vclMatrix float class initializer" ,{
    
    has_cpu_skip()
    
    vclD <- vclMatrix(D, type="float")
    
    expect_is(vclD, "fvclMatrix")
    expect_equal(vclD[], D, tolerance=1e-07, 
                 info="vcl float matrix elements not equivalent")
    expect_equal(dim(vclD), dim(D))
    expect_equal(ncol(vclD), ncol(D))
    expect_equal(nrow(vclD), nrow(D))
    expect_equal(typeof(vclD), "float")
})

test_that("CPU vclMatrix double class initializer" ,{
    
    has_cpu_skip()
    
    vclD <- vclMatrix(D)
    
    expect_is(vclD, "dvclMatrix")
    expect_equal(vclD[], D, tolerance=.Machine$double.eps ^ 0.5, 
                 info="vcl double matrix elements not equivalent")
    expect_equal(dim(vclD), dim(D))
    expect_equal(ncol(vclD), ncol(D))
    expect_equal(nrow(vclD), nrow(D))
    expect_equal(typeof(vclD), "double")
})

test_that("CPU vclMatrix vector initializers", {
    
    has_cpu_skip()
    
    v <- rnorm(10)
    vi <- seq.int(10)
    A <- matrix(v, nrow=5)
    Ai <- matrix(vi, nrow=2)
    err <- c(TRUE, FALSE)
    err2 <- c("hello", FALSE, 6)
    
    vclA <- vclMatrix(v, nrow=5, ncol=2, type="double")
    vclAi <- vclMatrix(vi, nrow=2, ncol=5)
    
    expect_equivalent(vclA[], A)
    expect_equal(dim(A), dim(vclA))
    expect_is(vclAi, "ivclMatrix")
    expect_equivalent(vclAi[], Ai)
    expect_equal(dim(Ai), dim(vclAi))
    expect_error(vclMatrix(err, nrow=1, ncol=2, type="double"))
    expect_error(vclMatrix(err, nrow=1, ncol=2))
    expect_error(vclMatrix(err2, nrow=1, ncol=3, type="double"))
})

test_that("CPU vclMatrix scalar initializers", {
    
    has_cpu_skip()
    
    v <- 3
    vi <- 4L
    A <- matrix(v, nrow=5, ncol=5)
    Ai <- matrix(vi, nrow=2, ncol=7)
    
    vclA <- vclMatrix(v, nrow=5, ncol=5, type="double")
    ivclA <- vclMatrix(vi, nrow=2, ncol=7, type="integer")
    
    expect_error(vclMatrix(v, nrow=5, ncol=5, type="integer"))
    
    expect_equivalent(vclA[], A,
                      "scalar double elements not equivalent")
    expect_equal(dim(A), dim(vclA),
                 "scalar double dimensions not equivalent")
    expect_is(ivclA, "ivclMatrix")
    expect_equivalent(ivclA[], Ai,
                      "scalar integer elements not equivalent")
    expect_equal(dim(Ai), dim(ivclA),
                 "scalar integer dimensions not equivalent")
})

options(gpuR.default.device = "gpu")
