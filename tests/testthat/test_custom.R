library(gpuR)
context("vclMatrix algebra")

library(Rcpp)
Sys.setenv(PKG_LIBS="-Lpath/to/opencl -lOpenCL -Wl,-rpath,path/to/opencl")
Sys.setenv(PKG_CPPFLAGS="-Ipath/to/gpuR/inst/include")

# Integer tests

test_that("vclMatrix Integer Matrix multiplication", {
    
    sourceCpp('../../gemm.cpp')
    
    print("arguments")
    print(formals(cpp_gpuMatrix_custom_igemm2))
    
    # setContext(2L)
    set.seed(123)
    
    ORDER <- 4
    
    # Base R objects
    Aint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
    Bint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
    
    igpuA <- vclMatrix(Aint, type="integer")
    igpuB <- vclMatrix(Bint, type="integer")
    igpuC <- vclMatrix(0L, 4, 4, type="integer")
    
    # basic_gemm(igpuA@address, igpuB@address, igpuC@address)
    
    file <- system.file("CL", "basic_gemm.cl", package = "gpuR")
    kernel <- readChar(file, file.info(file)$size)
    
    cpp_gpuMatrix_custom_igemm2(igpuA@address, TRUE, igpuB@address, TRUE,
                               igpuC@address, TRUE, kernel,
                               sqrt(256), igpuC@.context_index - 1L)
})

