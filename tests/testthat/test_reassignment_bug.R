library(gpuR)
context("Reassignment Bug Test")

# avoid downcast warnings for single precision
options(bigmemory.typecast.warning=FALSE)

## Check for big.matrix matrices

## Reassignemnt will currently cause crash
# print("Started big.matrix Crash Test")
# ORDER = 32
# 
# Aint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
# Bint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
# 
# igpuA <- gpuBigMatrix(Aint, type="integer")
# igpuB <- gpuBigMatrix(Bint, type="integer")
# 
# igpuC <- igpuA %*% igpuB
# 
# Aint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
# Bint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
# 
# igpuA <- gpuBigMatrix(Aint, type="integer")
# igpuB <- gpuBigMatrix(Bint, type="integer")
# 
# igpuC <- igpuA %*% igpuB
# print("PASSED CRASH!!!")


## Check for regular R matrices

# print("Started base R Crash Test")
# ORDER = 32
# 
# A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
# B <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
# 
# igpuA <- gpuMatrix(A, type="float")
# igpuB <- gpuMatrix(B, type="float")
# 
# igpuC <- igpuA %*% igpuB
# 
# A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
# B <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
# 
# igpuA <- gpuMatrix(A, type="float")
# igpuB <- gpuMatrix(B, type="float")
# 
# igpuC <- igpuA %*% igpuB
# 
# print("PASSED CRASH!!!")