Sys.setenv("R_TESTS" = "")
library(testthat)
library(gpuR)

test_check("gpuR")
