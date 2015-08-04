## ----setup, include=FALSE, cache=FALSE-----------------------------------
library(knitr)
opts_chunk$set(
concordance=TRUE
)

## ----install, eval = FALSE-----------------------------------------------
#  # Stable version
#  install.packages("gpuR")
#  
#  # Dev version
#  devtools::install_github("cdeterman/gpuR")

## ----matMult, eval=FALSE-------------------------------------------------
#  library("gpuR")
#  
#  # verify you have valid GPUs
#  detectGPUs()
#  
#  # create gpuMatrix and multiply
#  set.seed(123)
#  gpuA <- gpuMatrix(rnorm(16), nrow=4, ncol=4)
#  gpuB <- gpuA %*% gpuA

