library(gpuR)

ORDER <- 512

# Base R objects
A <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
B <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)

# GPU matrix objects
gpuA <- gpuMatrix(A)
gpuB <- gpuMatrix(B)

system.time(gpuA %*% gpuB)