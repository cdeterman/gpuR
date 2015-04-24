library(gpuR)
context("Matrix Utilities")

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
A <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
B <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
C <- A %*% B

# GPU matrix objects
gpuA <- gpuBigMatrix(A)
# gpuB <- gpuMatrix(B)

test_that("Matrix accession successful", {
    gpuA[,]
    expect_is(gpuA, "big.matrix")
#     expect_equivalent(dim(gpuC), dim(C), "matrix dimensions not equivalent")
#     expect_equivalent(gpuC@object, C, "matrix elements not equivalent")
})
