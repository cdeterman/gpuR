library(gpuR)
context("gpuMatrix Utility Functions")

set.seed(123)
A <- matrix(sample(seq.int(100), 16), 4)
D <- matrix(sample(rnorm(100), 16), 4)


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

test_that("gpuMatrix set elements", {
    
    has_gpu_skip()
    has_double_skip()
    
    dvec <- rnorm(ncol(D))
    ivec <- sample(seq.int(10), ncol(D))
    
    dgpu <- gpuMatrix(D)
    fgpu <- gpuMatrix(D, type="float")
    igpu <- gpuMatrix(A)

    dgpu[1,] <- dvec
    fgpu[1,] <- dvec
    igpu[1,] <- ivec
    D[1,] <- dvec
    A[1,] <- ivec

#     expect_equivalent(dgpu[,1], D[,1],
#                       info = "double column subset not equivalent")
#     expect_equal(fgpu[,1], D[,1], tolerance = 1e-07,
#                  info = "float column subset not equivalent ")
#     expect_equivalent(igpu[,1], A[,1],
#                       info = "integer column subset not equivalent")
    
    expect_equivalent(dgpu[1,], D[1,],
                      info = "double row set not equivalent")
    expect_equal(fgpu[1,], D[1,], tolerance = 1e-07,
                 info = "float row set not equivalent ")
    expect_equivalent(igpu[1,], A[1,],
                      info = "integer row set not equivalent")
#     
#     expect_equivalent(dgpu[1,2], D[1,2],
#                       info = "double element subset not equivalent")
#     expect_equal(fgpu[1,2], D[1,2], tolerance = 1e-07,
#                  info = "float element subset not equivalent ")
#     expect_equivalent(igpu[1,2], A[1,2],
#                       info = "integer element subset not equivalent")
})


