library(gpuR)
context("gpuMatrix Utility Functions")

set.seed(123)
A <- matrix(sample(seq.int(100), 100), 10)
D <- matrix(sample(rnorm(100), 100), 10)


test_that("gpuMatrix element access", {
    
    has_gpu_skip()
    has_double_skip()
    
    dgpu <- gpuMatrix(D)
    fgpu <- gpuMatrix(D, type="float")
    igpu <- gpuMatrix(A)
    
    expect_equivalent(dgpu[,1], D[,1],
                      info = "double column subset not equivalent")
    expect_equal(fgpu[,1], D[,1], tolerance = 1e-07,
                 info = "float column subset not equivalent ")
    expect_equivalent(igpu[,1], A[,1],
                      info = "integer column subset not equivalent")
    
    expect_equivalent(dgpu[1,], D[1,],
                      info = "double row subset not equivalent")
    expect_equal(fgpu[1,], D[1,], tolerance = 1e-07,
                 info = "float row subset not equivalent ")
    expect_equivalent(igpu[1,], A[1,],
                      info = "integer row subset not equivalent")
    
    expect_equivalent(dgpu[1,2], D[1,2],
                      info = "double element subset not equivalent")
    expect_equal(fgpu[1,2], D[1,2], tolerance = 1e-07,
                 info = "float element subset not equivalent ")
    expect_equivalent(igpu[1,2], A[1,2],
                      info = "integer element subset not equivalent")
})

test_that("gpuMatrix set column access", {
    
    has_gpu_skip()
    has_double_skip()
    
    gpuA <- gpuMatrix(A)
    gpuD <- gpuMatrix(D)
    gpuF <- gpuMatrix(D, type = "float")
    gpuB <- gpuD
    
    icolvec <- sample(seq.int(10), 10)
    colvec <- rnorm(10)
    
    gpuA[,1] <- icolvec
    gpuD[,1] <- colvec
    gpuF[,1] <- colvec
    
    A[,1] <- icolvec
    D[,1] <- colvec
    
    expect_equivalent(gpuD[,1], colvec,
                      info = "updated dgpuMatrix column not equivalent")
    expect_equivalent(gpuD[], D,
                      info = "updated dgpuMatrix not equivalent")
    expect_equivalent(gpuB[], D, 
                      info = "updated dgpuMatrix column not reflected in 'copy'")
    expect_equal(gpuF[,1], colvec, tolerance=1e-07,
                 info = "updated fgpuMatrix column not equivalent")
    expect_equal(gpuF[], D, tolerance=1e-07,
                 info = "updated fgpuMatrix not equivalent")
    expect_equivalent(gpuA[,1], icolvec,
                      info = "updated igpuMatrix column not equivalent")
    expect_equivalent(gpuA[], A,
                      info = "updated igpuMatrix not equivalent")
    expect_error(gpuA[,11] <- icolvec,
                 info = "no error when index greater than dims")
    expect_error(gpuD[,1] <- rnorm(12),
                 info = "no error when vector larger than number of rows")
})

test_that("gpuMatrix set row access", {
    
    has_gpu_skip()
    has_double_skip()
    
    gpuA <- gpuMatrix(A)
    gpuD <- gpuMatrix(D)
    gpuF <- gpuMatrix(D, type = "float")
    gpuB <- gpuD
    
    icolvec <- sample(seq.int(10), 10)
    colvec <- rnorm(10)
    
    gpuA[1,] <- icolvec
    gpuD[1,] <- colvec
    gpuF[1,] <- colvec
    
    A[1,] <- icolvec
    D[1,] <- colvec
    
    expect_equivalent(gpuD[1,], colvec,
                      info = "updated dgpuMatrix row not equivalent")
    expect_equivalent(gpuD[], D,
                      info = "updated dgpuMatrix not equivalent")
    expect_equivalent(gpuB[], D, 
                      info = "updated dgpuMatrix row not reflected in 'copy'")
    expect_equal(gpuF[1,], colvec, tolerance=1e-07,
                 info = "updated fgpuMatrix row not equivalent")
    expect_equal(gpuF[], D, tolerance=1e-07,
                 info = "updated fgpuMatrix not equivalent")
    expect_equivalent(gpuA[1,], icolvec,
                      info = "updated igpuMatrix row not equivalent")
    expect_equivalent(gpuA[], A,
                      info = "updated igpuMatrix not equivalent")
    expect_error(gpuA[11,] <- icolvec,
                 info = "no error when index greater than dims")
    expect_error(gpuD[1,] <- rnorm(12),
                 info = "no error when vector larger than number of rows")
})

test_that("gpuMatrix set element access", {
    
    has_gpu_skip()
    has_double_skip()
    
    gpuA <- gpuMatrix(A)
    gpuD <- gpuMatrix(D)
    gpuF <- gpuMatrix(D, type = "float")
    gpuB <- gpuD
    
    int <- sample(seq.int(10), 1)
    float <- rnorm(1)
    
    gpuA[1,3] <- int
    gpuD[1,3] <- float
    gpuF[1,3] <- float
    
    A[1,3] <- int
    D[1,3] <- float
    
    expect_equivalent(gpuD[1,3], float,
                      info = "updated dgpuMatrix element not equivalent")
    expect_equivalent(gpuD[], D,
                      info = "updated dgpuMatrix not equivalent")
    expect_equivalent(gpuB[], D, 
                      info = "updated dgpuMatrix elemnent not reflected in 'copy'")
    expect_equal(gpuF[1,3], float, tolerance=1e-07,
                 info = "updated fgpuMatrix element not equivalent")
    expect_equal(gpuF[], D, tolerance=1e-07,
                 info = "updated fgpuMatrix not equivalent")
    expect_equivalent(gpuA[1,3], int,
                      info = "updated igpuMatrix element not equivalent")
    expect_equivalent(gpuA[], A,
                      info = "updated igpuMatrix not equivalent")
    expect_error(gpuA[11,3] <- int,
                 info = "no error when index greater than dims")
    expect_error(gpuD[1,3] <- rnorm(12),
                 info = "no error when assigned vector to element")
})

test_that("gpuMatrix confirm print doesn't error", {
    
    has_gpu_skip()
   
    dgpu <- gpuMatrix(D, type="float")

    expect_that(print(dgpu), prints_text("Source: gpuR Matrix"))
})