library(gpuR)
context("CPU vclVector Utility Functions")

# set option to use CPU instead of GPU
options(gpuR.default.device.type = "cpu")

set.seed(123)
ORDER <- 100
A <- sample(seq.int(10), ORDER, replace = TRUE)
D <- rnorm(ORDER)

test_that("integer vclVector length method successful", {
    
    has_cpu_skip()
    
    gpuA <- vclVector(A)
    
    s <- length(gpuA)
    
    expect_true(s == ORDER)
})

test_that("float vclVector length method successful", {
    
    has_cpu_skip()
    
    gpuA <- vclVector(A, type="float")
    
    s <- length(gpuA)
    
    expect_true(s == ORDER)
})

test_that("double vclVector length method successful", {
    
    has_cpu_skip()
    
    gpuA <- vclVector(A, type="double")
    
    s <- length(gpuA)
    
    expect_true(s == ORDER)
})

test_that("vclVector accession method successful", {
    
    has_cpu_skip()
    
    gpuA <- vclVector(A)
    gpuF <- vclVector(D, type="float")
    
    gi <- gpuA[2]
    i <- A[2]
    gs <- gpuF[2]
    s <- D[2]
    
    expect_equivalent(gi, i, info = "ivclVector element access not correct")
    expect_equal(gs, s, tolerance = 1e-07, 
                 info = "fvclVector element access not correct")
    expect_error(gpuA[101], info = "no error when outside vclVector size")
})

test_that("dvclVector accession method successful", {
    
    has_cpu_skip()
    
    gpuD <- vclVector(D)
    
    gs <- gpuD[2]
    s <- D[2]
    
    expect_equivalent(gs, s, info = "dvclVector element access not correct")
})

test_that("vclVector set accession method successful", {

    has_cpu_skip()
    
    Ai <- sample(seq.int(10), 10, replace = TRUE)
    
    gpuA <- vclVector(Ai)
    gpuF <- vclVector(D, type="float")
    
    int = 13L
    float = rnorm(1)
    
    gpuA[2] <- int
    Ai[2] <- int
    gpuF[2] <- float
    D[2] <- float
    
    expect_equivalent(gpuA[], Ai, 
                      info = "ivclVector set element access not correct")
    expect_equal(gpuF[], D, tolerance = 1e-07, 
                 info = "fvclVector set element access not correct")
    expect_error(gpuA[101] <- 42, 
                 info = "no error when set outside ivclVector size")
    expect_error(gpuF[101] <- 42.42, 
                 info = "no error when set outside fvclVector size")
})

test_that("dvclVector set accession method successful", {
    
    has_cpu_skip()
    
    gpuD <- vclVector(D)
    
    float = rnorm(1)
    
    gpuD[2] <- float
    D[2] <- float
    
    expect_equivalent(gpuD[], D, 
                      info = "dvclVector set element access not correct")
    expect_error(gpuD[101] <- 42.42, 
                 info = "no error when set outside dvclVector size")
})


options(gpuR.default.device.type = "gpu")
