library(gpuR)
context("CPU vclMatrix eigen decomposition")

# set option to use CPU instead of GPU
options(gpuR.default.device.type = "cpu")

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
X <- A %*% t(A)

E <- eigen(X)
Q <- E$vectors
V <- E$values

nE <- eigen(A)
nQ <- nE$vectors
nV <- nE$values


test_that("vclMatrix Symmetric Single Precision Matrix Eigen Decomposition",
{    
    has_cpu_skip()
    
    fgpuX <- vclMatrix(X, type="float")
    
    E <- eigen(fgpuX, symmetric=TRUE)
    
    # need to reorder so it matches R output
    ord <- order(E$values[], decreasing = TRUE)
    
    expect_is(E, "list")
    expect_equal(E$values[][ord], V, tolerance=1e-06, 
                 info="float eigenvalues not equivalent")  
    
    # need abs as some signs are opposite (not important with eigenvectors)
    expect_equal(abs(E$vectors[][,ord]), abs(Q), tolerance=1e-06, 
                 info="float eigenvectors not equivalent")  
})

test_that("vclMatrix Symmetric Double Precision Matrix Eigen Decomposition", 
{    
    has_cpu_skip()
    
    fgpuX <- vclMatrix(X, type="double")
    
    E <- eigen(fgpuX, symmetric=TRUE)
    
    # need to reorder so it matches R output
    ord <- order(E$values[], decreasing = TRUE)
    
    expect_is(E, "list")
    expect_equal(E$values[][ord], V, tolerance=1e-06, 
                 info="float eigenvalues not equivalent")  
    
    # need abs as some signs are opposite (not important with eigenvectors)
    expect_equal(abs(E$vectors[][,ord]), abs(Q), tolerance=1e-06, 
                 info="float eigenvectors not equivalent")  
})

# test_that("vclMatrix Non-Symmetric Single Precision Matrix Eigen Decomposition",
# {
#     
#     has_gpu_skip()
#     
#     fgpuX <- vclMatrix(A, type="float")
#     
#     E <- eigen(fgpuX)
#     
#     # need to reorder so it matches R output
#     ord <- order(E$values[], decreasing = TRUE)
#     
#     print(nV)
#     print(E$values[][ord])
#     
#     expect_is(E, "list")
#     expect_equal(E$values[][ord], nV, tolerance=1e-06, 
#                  info="float eigenvalues not equivalent")  
#     
#     # need abs as some signs are opposite (not important with eigenvectors)
#     expect_equal(abs(E$vectors[][,ord]), abs(nQ), tolerance=1e-06, 
#                  info="float eigenvectors not equivalent")  
# })
# 
# test_that("vclMatrix Non-Symmetric Double Precision Matrix Eigen Decomposition", 
# {
#     
#     has_gpu_skip()
#     has_double_skip()
#     
#     fgpuX <- vclMatrix(A, type="double")
#     
#     E <- eigen(fgpuX)
#     
#     # need to reorder so it matches R output
#     ord <- order(E$values[], decreasing = TRUE)
#     
#     expect_is(E, "list")
#     expect_equal(E$values[][ord], nV, 
#                  tolerance=.Machine$double.eps ^ 0.5, 
#                  info="float eigenvalues not equivalent")  
#     
#     # need abs as some signs are opposite (not important with eigenvectors)
#     expect_equal(abs(E$vectors[][,ord]), abs(nQ), 
#                  tolerance=.Machine$double.eps ^ 0.5, 
#                  info="float eigenvectors not equivalent")  
# })

options(gpuR.default.device.type = "gpu")
