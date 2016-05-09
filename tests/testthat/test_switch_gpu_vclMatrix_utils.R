library(gpuR)
context("Switching GPU vclMatrix Utility Functions")

set.seed(123)
A <- matrix(seq.int(100), 10)
D <- matrix(rnorm(100), 10)


test_that("Switching GPU vclMatrix get element access", {
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    setContext(2L)
    
    dgpu <- vclMatrix(D)
    fgpu <- vclMatrix(D, type="float")
    igpu <- vclMatrix(A)
    
    setContext(1L)
    
    expect_equal(dgpu@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(fgpu@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(igpu@.context_index, 2L,
                 info = "context index not assigned properly")
    
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

    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix set column access", {
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    setContext(2L)
    
    gpuA <- vclMatrix(A)
    gpuD <- vclMatrix(D)
    gpuF <- vclMatrix(D, type = "float")
    gpuB <- gpuD
    
    setContext(1L)
    
    icolvec <- sample(seq.int(10), 10)
    colvec <- rnorm(10)
    
    gpuA[,1] <- icolvec
    gpuD[,1] <- colvec
    gpuF[,1] <- colvec
    
    A[,1] <- icolvec
    D[,1] <- colvec
    
    
    expect_equal(gpuA@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(gpuD@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(gpuF@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(gpuB@.context_index, 2L,
                 info = "context index not assigned properly")
    
    expect_equivalent(gpuD[,1], colvec,
                      info = "updated dvclMatrix column not equivalent")
    expect_equivalent(gpuD[], D,
                      info = "updated dvclMatrix not equivalent")
    expect_equivalent(gpuB[], D, 
                      info = "updated dvclMatrix column not reflected in 'copy'")
    expect_equal(gpuF[,1], colvec, tolerance=1e-07,
                 info = "updated fvclMatrix column not equivalent")
    expect_equal(gpuF[], D, tolerance=1e-07,
                 info = "updated fvclMatrix not equivalent")
    expect_equivalent(gpuA[,1], icolvec,
                      info = "updated ivclMatrix column not equivalent")
    expect_equivalent(gpuA[], A,
                      info = "updated ivclMatrix not equivalent")
    expect_error(gpuA[,11] <- icolvec,
                 info = "no error when index greater than dims")
    expect_error(gpuD[,1] <- rnorm(12),
                 info = "no error when vector larger than number of rows")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix set row access", {
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    setContext(2L)
    
    gpuA <- vclMatrix(A)
    gpuD <- vclMatrix(D)
    gpuF <- vclMatrix(D, type = "float")
    gpuB <- gpuD
    
    setContext(1L)
    
    icolvec <- sample(seq.int(10), 10)
    colvec <- rnorm(10)
    
    gpuA[1,] <- icolvec
    gpuD[1,] <- colvec
    gpuF[1,] <- colvec
    
    A[1,] <- icolvec
    D[1,] <- colvec
    
    expect_equal(gpuA@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(gpuD@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(gpuF@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(gpuB@.context_index, 2L,
                 info = "context index not assigned properly")
    
    expect_equivalent(gpuD[1,], colvec,
                      info = "updated dvclMatrix row not equivalent")
    expect_equivalent(gpuD[], D,
                      info = "updated dvclMatrix not equivalent")
    expect_equivalent(gpuB[], D, 
                      info = "updated dvclMatrix row not reflected in 'copy'")
    expect_equal(gpuF[1,], colvec, tolerance=1e-07,
                 info = "updated fvclMatrix row not equivalent")
    expect_equal(gpuF[], D, tolerance=1e-07,
                 info = "updated fvclMatrix not equivalent")
    expect_equivalent(gpuA[1,], icolvec,
                      info = "updated ivclMatrix row not equivalent")
    expect_equivalent(gpuA[], A,
                      info = "updated ivclMatrix not equivalent")
    expect_error(gpuA[11,] <- icolvec,
                 info = "no error when index greater than dims")
    expect_error(gpuD[1,] <- rnorm(12),
                 info = "no error when vector larger than number of rows")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix set element access", {
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    setContext(2L)
    
    gpuA <- vclMatrix(A)
    gpuD <- vclMatrix(D)
    gpuF <- vclMatrix(D, type = "float")
    gpuB <- gpuD
    
    setContext(1L)
    
    int <- sample(seq.int(10), 1)
    float <- rnorm(1)
    
    gpuA[1,3] <- int
    gpuD[1,3] <- float
    gpuF[1,3] <- float
    
    A[1,3] <- int
    D[1,3] <- float
    
    expect_equal(gpuA@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(gpuD@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(gpuF@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(gpuB@.context_index, 2L,
                 info = "context index not assigned properly")
    
    expect_equivalent(gpuD[1,3], float,
                      info = "updated dvclMatrix element not equivalent")
    expect_equivalent(gpuD[], D,
                      info = "updated dvclMatrix not equivalent")
    expect_equivalent(gpuB[], D, 
                      info = "updated dvclMatrix elemnent not reflected in 'copy'")

    expect_equal(gpuF[1,3], float, tolerance=1e-07,
                 info = "updated fvclMatrix element not equivalent")
    expect_equal(gpuF[], D, tolerance=1e-07,
                 info = "updated fvclMatrix not equivalent")

    expect_equivalent(gpuA[1,3], int,
                      info = "updated ivclMatrix element not equivalent")
    expect_equivalent(gpuA[], A,
                      info = "updated ivclMatrix not equivalent")

    expect_error(gpuA[11,3] <- int,
                 info = "no error when index greater than dims")
    expect_error(gpuD[1,3] <- rnorm(12),
                 info = "no error when assigned vector to element")

    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})
