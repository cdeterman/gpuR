library(gpuR)
context("CPU gpuMatrix Utility Functions")

# set option to use CPU instead of GPU
options(gpuR.default.device = "cpu")

set.seed(123)
A <- matrix(seq.int(100), 10)
D <- matrix(rnorm(100), 10)


test_that("gpuMatrix element access", {
    
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

options(gpuR.default.device = "gpu")
