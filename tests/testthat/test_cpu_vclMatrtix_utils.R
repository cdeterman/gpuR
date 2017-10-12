library(gpuR)
context("CPU vclMatrix Utility Functions")

current_context <- set_device_context("cpu")

set.seed(123)
A <- matrix(seq.int(100), 10)
D <- matrix(rnorm(100), 10)
D2 <- matrix(rnorm(100), 10)

cnames <- paste0("V", seq(10))

test_that("CPU vclMatrix get element access", {
    has_cpu_skip()
    
    dgpu <- vclMatrix(D)
    fgpu <- vclMatrix(D, type="float")
    igpu <- vclMatrix(A)
    
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
    
    expect_equivalent(igpu[1:4,1:4], A[1:4,1:4],
                      info = "row & column subsets of ivclMatrix not equivalent")
    expect_equal(fgpu[1:4,1:4], D[1:4,1:4], tolerance = 1e-07,
                 info = "row & column subsets of fvclMatrix not equivalent")
    expect_equivalent(dgpu[1:4,1:4], D[1:4,1:4],
                      info = "row & column subsets of dvclMatrix not equivalent")
})

test_that("CPU vclMatrix set column access", {
    has_cpu_skip()
    
    gpuA <- vclMatrix(A)
    gpuD <- vclMatrix(D)
    gpuF <- vclMatrix(D, type = "float")
    gpuB <- gpuD
    
    icolvec <- sample(seq.int(10), 10)
    colvec <- rnorm(10)
    
    gpuA[,1] <- icolvec
    gpuD[,1] <- colvec
    gpuF[,1] <- colvec
    
    A[,1] <- icolvec
    D[,1] <- colvec
    
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
    
    expect_equivalent(gpuA[,1:4], A[,1:4],
                      info = "column subsets of ivclMatrix not equivalent")
    expect_equivalent(gpuD[,1:4], D[,1:4],
                      info = "column subsets of fvclMatrix not equivalent")
    expect_equal(gpuF[,1:4], D[,1:4], tolerance = 1e-07,
                 info = "column subsets of dvclMatrix not equivalent")
})

test_that("CPU vclMatrix set row access", {
    has_cpu_skip()
    
    gpuA <- vclMatrix(A)
    gpuD <- vclMatrix(D)
    gpuF <- vclMatrix(D, type = "float")
    gpuB <- gpuD
    
    icolvec <- sample(seq.int(10), 10)
    colvec <- rnorm(10)
    
    gpuA[1,] <- icolvec
    gpuD[1,] <- colvec
    gpuF[1,] <- colvec
    
    A[1,] <- icolvec
    D[1,] <- colvec
    
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
    
    expect_equivalent(gpuA[1:4,], A[1:4,],
                      info = "row subsets of ivclMatrix not equivalent")
    expect_equivalent(gpuD[1:4,], D[1:4,], 
                      info = "row subsets of fvclMatrix not equivalent")
    expect_equal(gpuF[1:4,], D[1:4,], tolerance = 1e-07,
                 info = "row subsets of dvclMatrix not equivalent")
})

test_that("CPU vclMatrix set element access", {
    has_cpu_skip()
    
    gpuA <- vclMatrix(A)
    gpuD <- vclMatrix(D)
    gpuF <- vclMatrix(D, type = "float")
    gpuB <- gpuD
    
    int <- sample(seq.int(10), 1)
    float <- rnorm(1)
    
    gpuA[1,3] <- int
    gpuD[1,3] <- float
    gpuF[1,3] <- float
    
    A[1,3] <- int
    D[1,3] <- float
    
    D[c(6,10)] <- 0
    gpuD[c(6,10)] <- 0
    gpuF[c(6,10)] <- 0
    
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
    expect_equivalent(gpuD[c(6,10)], D[c(6,10)],
                      info = "double non-contiguous subset not equivalent")
})

test_that("CPU vclMatrix as.matrix method", {
    
    has_cpu_skip()
    
    dgpu <- vclMatrix(D)
    fgpu <- vclMatrix(D, type="float")
    igpu <- vclMatrix(A)
    
    expect_equal(as.matrix(dgpu), D,
                      info = "double as.matrix not equal")
    expect_equal(as.matrix(fgpu), D,
                      info = "float as.matrix not equal",
                      tolerance = 1e-07)
    expect_equal(as.matrix(dgpu), D,
                      info = "integer as.matrix not equal")
    
    
    expect_is(as.matrix(dgpu), 'matrix',
              info = "double as.matrix not producing 'matrix' class")
    expect_is(as.matrix(fgpu), 'matrix',
              info = "float as.matrix not producing 'matrix' class")
    expect_is(as.matrix(igpu), 'matrix',
              info = "integer as.matrix not producing 'matrix' class")
})

test_that("CPU vclMatrix colnames methods", {
    
    has_cpu_skip()
    
    fgpu <- vclMatrix(D, type="float")
    igpu <- vclMatrix(A)
    
    expect_null(colnames(fgpu), 
                info = "float colnames should return NULL before assignment")
    expect_null(colnames(igpu), 
                info = "integer colnames should return NULL before assignment")
    
    colnames(fgpu) <- cnames
    colnames(igpu) <- cnames
    
    expect_equal(colnames(fgpu), cnames,
                 info = "float colnames don't reflect assigned names")
    expect_equal(colnames(igpu), cnames,
                 info = "integer colnames don't reflect assigned names")
    
    # Double tests
    
    dgpu <- vclMatrix(D)
    
    expect_null(colnames(dgpu), 
                info = "double colnames should return NULL before assignment")
    
    colnames(dgpu) <- cnames
    
    expect_equal(colnames(dgpu), cnames,
                 info = "double colnames don't reflect assigned names")
})

test_that("CPU vclMatrix set matrix access", {
    
    has_cpu_skip()
    
    gpuA <- vclMatrix(D)
    gpuF <- vclMatrix(D, type = "float")
    
    gpuA[] <- D2
    gpuF[] <- D2
    
    expect_equivalent(gpuA[], D2,
                      info = "updated dvclMatrix not equivalent to assigned base matrix")
    
    expect_equal(gpuF[], D2, tolerance=1e-07,
                 info = "updated fvclMatrix not equivalent to assigned base matrix")
    
})

test_that("CPU vclMatrix set vclMatrix access", {
    
    has_cpu_skip()
    
    gpuA <- vclMatrix(D)
    gpuD <- vclMatrix(D2)
    gpuF <- vclMatrix(D, type = "float")
    gpuDF <- vclMatrix(D2, type = "float")
    
    gpuA[] <- gpuD
    gpuF[] <- gpuDF
    
    expect_equivalent(gpuA[], gpuD[], 
                      info = "updated dvclMatrix not equivalent to assigned vclMatrix")
    expect_equal(gpuF[], gpuDF[], tolerance=1e-07,
                 info = "updated fvclMatrix not equivalent to assigned base vclMatrix")
})

setContext(current_context)
