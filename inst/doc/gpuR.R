## ----setup, include=FALSE, cache=FALSE-----------------------------------
library(knitr)
opts_chunk$set(
concordance=TRUE
)

## ----install, eval = FALSE-----------------------------------------------
#  # Stable version
#  install.packages("gpuR")

## ----installDev, eval = FALSE--------------------------------------------
#  # Dev version
#  devtools::install_github("cdeterman/gpuR", ref = "develop")
#  
#  # Note this may require install of the RViennaCL from my github
#  # if updates have been made
#  #devtools::install_github("cdeterman/RViennaCL")

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

## ----matBlock, eval=FALSE------------------------------------------------
#  
#  # create gpuMatrix
#  set.seed(123)
#  gpuA <- gpuMatrix(rnorm(16), nrow=4, ncol=4)
#  
#  # create block omitting the 1st row
#  gpuB <- block(gpuA,
#                rowStart = 2L, rowEnd = 4L,
#                colStart = 1L, colEnd = 4L)

