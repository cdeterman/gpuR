library(gpuR)
context("Matrix algebra")

# avoid downcast warnings for single precision
options(bigmemory.typecast.warning=FALSE)

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
Aint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
Bint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
B <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)

# print("R A matrix")
# print(Aint)
# print("R B matrix")
# print(Bint)

Cint <- Aint %*% Bint
C <- A %*% B

# GPU matrix objects
gpuA <- gpuMatrix(A, type="float")
gpuB <- gpuMatrix(B, type="float")

igpuA <- gpuMatrix(Aint, type="integer")
igpuB <- gpuMatrix(Bint, type="integer")

test_that("Single Precision Matrix multiplication successful", {
    
#     print("matrices declared")
#     cpp_cudaGEMM()
    
    gpuC <- gpuA %*% gpuB
    igpuC <- igpuA %*% igpuB
    
#     print("r default")
#     print(C)
# #     print(C[1:2,1:5])
#     print("gpu")
# #     print(head(gpuC@object, 1))
# #     print(gpuC[1:2,1:5])
# #     print(gpuC@object)
#     print(gpuC[,])

#     print("r default")
#     print(Cint)
#     print("gpu")
#     print(igpuC[,])

    expect_is(gpuC, "gpuMatrix")
    expect_is(igpuC, "igpuMatrix")
    expect_equivalent(dim(gpuC), dim(C), 
                      "float matrix dimensions not equivalent")
    expect_equivalent(dim(igpuC), dim(Cint), 
                      "integer matrix dimensions not equivalent")
    expect_equal(gpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")
    expect_equal(igpuC[,], Cint, 
                 info="integer matrix elements not equivalent")
})

test_that("Matrix Reassignment doesn't cause crash", {
    ORDER = 32
    
    Aint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
    Bint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
    
    igpuA <- gpuMatrix(Aint, type="integer")
    igpuB <- gpuMatrix(Bint, type="integer")
    
    igpuC <- igpuA %*% igpuB
    
    ORDER = 64
    
    Aint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
    Bint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
    
    igpuA <- gpuMatrix(Aint, type="integer")
    igpuB <- gpuMatrix(Bint, type="integer")
    
    igpuC <- igpuA %*% igpuB
    
    expect_true(TRUE)
})

# test_that("Matrix additonal successful", {
#     gpuA <- as.gpuMatrix(A)
#     gpuB <- as.gpuMatrix(B)
#     
#     # R default
#     C <- A + B
#     
#     # manual call
#     #gpuC <- gpu_vec_add(A, B)
#     
#     # generic call
#     gpuC <- gpuA + gpuB
#     
#     expect_equivalent(gpuC@object, C)
# })
# 
# test_that("Matrix subtraction successful", {
#     gpuA <- as.gpuMatrix(A)
#     gpuB <- as.gpuMatrix(B)
#     
#     # R default
#     C <- A - B
#     
#     # generic call
#     gpuC <- gpuA - gpuB
#     
#     expect_equivalent(gpuC@object, C)
# })
