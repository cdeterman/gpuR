library(gpuR)
context("Switching GPU vclMatrix eigen decomposition")

# set seed
set.seed(123)

ORDER <- 10

# Base R objects
A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
X <- tcrossprod(A)

E <- eigen(X)
Q <- E$vectors
V <- E$values

nE <- eigen(A)
nQ <- nE$vectors
nV <- nE$values


test_that("Switching GPU vclMatrix Symmetric Single Precision Matrix Eigen Decomposition",
{
    skip("switching contexts not function in ViennaCL yet")
    
    has_multiple_gpu_skip()
    
    setContext(2L)
    
    fgpuX <- vclMatrix(X, type="float")
    
    setContext(1L)
    
    E <- eigen(fgpuX, symmetric=TRUE)
    
    # need to reorder so it matches R output
    ord <- order(E$values[], decreasing = TRUE)
    
    expect_is(E, "list")
    expect_equal(E$values[][ord], V, tolerance=1e-06, 
                 info="float eigenvalues not equivalent")  
    
    # need abs as some signs are opposite (not important with eigenvectors)
    expect_equal(abs(E$vectors[][,ord]), abs(Q), tolerance=1e-05, 
                 info="float eigenvectors not equivalent")  
    
    # make sure X not overwritten
    expect_equal(fgpuX[], X, tolerance=1e-06, 
                 info="float source matrices not equivalent") 
    
    # context checks
    expect_equal(E$values@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(E$vectors@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Symmetric Double Precision Matrix Eigen Decomposition", 
{
    skip("switching contexts not function in ViennaCL yet")
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    setContext(2L)
    
    fgpuX <- vclMatrix(X, type="double")
    
    setContext(1L)
    
    E <- eigen(fgpuX, symmetric=TRUE)     
    
    # need to reorder so it matches R output
    ord <- order(E$values[], decreasing = TRUE)
    
    expect_is(E, "list")
    expect_equal(E$values[][ord], V, tolerance=.Machine$double.eps ^ 0.5, 
                 info="float eigenvalues not equivalent")  
    
    # need abs as some signs are opposite (not important with eigenvectors)
    expect_equal(abs(E$vectors[][,ord]), abs(Q), tolerance=.Machine$double.eps ^ 0.5, 
                 info="float eigenvectors not equivalent")  
    
    # make sure X not overwritten
    expect_equal(fgpuX[], X, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double source matrices not equivalent") 
    
    # context checks
    expect_equal(E$values@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(E$vectors@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

# test_that("Switching GPU vclMatrix Non-Symmetric Single Precision Matrix Eigen Decomposition",
# {
#     
#     has_multiple_gpu_skip()
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
# test_that("Switching GPU vclMatrix Non-Symmetric Double Precision Matrix Eigen Decomposition", 
# {
#     
#     has_multiple_gpu_skip()
#     has_multiple_double_skip()
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
