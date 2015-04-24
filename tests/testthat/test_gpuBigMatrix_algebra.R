library(gpuR)
context("gpuBigMatrix algebra")

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



# print("matrix A")
# print(A)
# print("matrix B")
# print(B)

test_that("gpuBigMatrix Integer Matrix Multiplication successful", {
    
    igpuA <- gpuBigMatrix(Aint, type="integer")
    igpuB <- gpuBigMatrix(Bint, type="integer")
    
    igpuC <- igpuA %*% igpuB
    
    expect_is(igpuC, "igpuBigMatrix")
    expect_equivalent(dim(igpuC), dim(Cint), 
                      "integer matrix dimensions not equivalent")
    expect_equal(igpuC[,], Cint, 
                 info="integer matrix elements not equivalent")
#     rm(igpuC)
#     gc()
})

test_that("gpuBigMatrix Single Precision Matrix multiplication successful", {
    
    # GPU matrix objects
    fgpuA <- gpuBigMatrix(A, type="float")
    fgpuB <- gpuBigMatrix(B, type="float")
    
    fgpuC <- fgpuA %*% fgpuB

    expect_is(fgpuC, "fgpuBigMatrix")
    expect_equivalent(dim(fgpuC), dim(C), 
                      "float matrix dimensions not equivalent")
    expect_equal(fgpuC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")
})



# test_that("Matrix additonal successful", {
#     gpuA <- as.gpuBigMatrix(A)
#     gpuB <- as.gpuBigMatrix(B)
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
#     gpuA <- as.gpuBigMatrix(A)
#     gpuB <- as.gpuBigMatrix(B)
#     
#     # R default
#     C <- A - B
#     
#     # generic call
#     gpuC <- gpuA - gpuB
#     
#     expect_equivalent(gpuC@object, C)
# })
