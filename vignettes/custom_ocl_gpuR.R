## ----setup, include=FALSE, cache=FALSE-----------------------------------
library(knitr)
opts_chunk$set(
concordance=TRUE
)

## ----kernel_example, eval = FALSE----------------------------------------
#  __kernel void SAXPY(__global float* x, __global float* y, float a)
#  {
#      const int i = get_global_id(0);
#  
#      y [i] += a * x [i];
#  }

## ----cl_setup, eval=FALSE------------------------------------------------
#  cl_args <- setup_opencl(objects = c("vclVector", "vclVector", "scalar"),
#                          intents = c("IN", "OUT", "IN"),
#                          queues = list("SAXPY", "SAXPY", "SAXPY"),
#                          kernel_maps = c("x", "y", "a"))

## ----compile, eval=FALSE-------------------------------------------------
#  custom_opencl("saxpy.cl", cl_args, "float")

## ----example, eval=FALSE-------------------------------------------------
#  
#  a <- rnorm(16)
#  b <- rnorm(16)
#  gpuA <- vclVector(a, type = "float")
#  gpuB <- vclVector(b, type = "float")
#  scalar <- 2
#  
#  # apply custom function
#  # equivalent to - scalar*a + b
#  saxpy(gpuA, gpuB, scalar)

