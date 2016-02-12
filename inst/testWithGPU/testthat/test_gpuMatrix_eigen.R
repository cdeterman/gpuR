library(gpuR)
context("gpuMatrix eigen decomposition")

# set seed
set.seed(123)

ORDER <- 10

# Base R objects
A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
X <- A %*% t(A)

E <- eigen(X)
Q <- E$vectors
V <- E$values

nE <- eigen(A)
nQ <- nE$vectors
nV <- nE$values


test_that("gpuMatrix Symmetric Single Precision Matrix Eigen Decomposition",
{
    
    has_gpu_skip()
    
    fgpuX <- gpuMatrix(X, type="float")
    
    E <- eigen(fgpuX, symmetric=TRUE)
    
    # need to reorder so it matches R output
    ord <- order(E$values[], decreasing = TRUE)
    
    expect_is(E, "list")
    expect_equal(E$values[][ord], V, tolerance=1e-06, 
                 info="float eigenvalues not equivalent")  
    
    # need abs as some signs are opposite (not important with eigenvectors)
    expect_equal(abs(E$vectors[][,ord]), abs(Q), tolerance=1e-05, 
                 info="float eigenvectors not equivalent")  
})

test_that("gpuMatrix Symmetric Double Precision Matrix Eigen Decomposition", 
{
    
    has_gpu_skip()
    has_double_skip()
    
    fgpuX <- gpuMatrix(X, type="double")
    
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

# test_that("gpuMatrix Non-Symmetric Single Precision Matrix Eigen Decomposition",
# {
#     
#     has_gpu_skip()
#     
#     fgpuX <- gpuMatrix(A, type="float")
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
# test_that("gpuMatrix Non-Symmetric Double Precision Matrix Eigen Decomposition", 
# {
#     
#     has_gpu_skip()
#     has_double_skip()
#     
#     fgpuX <- gpuMatrix(A, type="double")
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


