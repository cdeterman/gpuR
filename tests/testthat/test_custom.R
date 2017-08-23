library(gpuR)
context("Custom OpenCL")

current_context <- set_device_context("cpu")

library(Rcpp)
Sys.setenv(PKG_LIBS="-LC:/Users/y66534/Documents/R_projects/gpuR/inst/include/loader/x64 -lOpenCL -Wl,-rpath,C:/Users/y66534/Documents/R_projects/gpuR/inst/include/loader/x64")
Sys.setenv(PKG_CPPFLAGS="-IC:/Users/y66534/Documents/R_projects/gpuR/inst/include")

# setContext(2L)
set.seed(123)

ORDER <- 4

# Base R objects
Aint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
Bint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)

igpuA <- vclMatrix(Aint, type="integer")
igpuB <- vclMatrix(Bint, type="integer")
igpuC <- vclMatrix(0L, 4, 4, type="integer")

# Integer tests

test_that("Custom OpenCL GEMM Kernel", {
    
    has_gpu_skip()
    
    Cint <- Aint %*% Bint
    
    kernel <- system.file("CL", "basic_gemm.cl", package = "gpuR")
    
    cl_args <- setup_opencl(c("vclMatrix", "vclMatrix", "vclMatrix"), 
                            toupper(c("in", "in", "inout")), 
                            list("iMatMult","iMatMult","iMatMult"), 
                            list("A", "B", "C"))
    
    custom_opencl(kernel, cl_args, type = "integer")
    
    basic_gemm(igpuA, igpuB, igpuC)
    
    expect_equivalent(igpuC[,], Cint,
                      info="integer matrix elements not equivalent")
    
})

setContext(current_context)