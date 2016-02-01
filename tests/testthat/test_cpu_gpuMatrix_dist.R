library(gpuR)
context("CPU gpuMatrix Distance Computations")

# set option to use CPU instead of GPU
options(gpuR.default.device.type = "cpu")

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)

D <- as.matrix(dist(A))

test_that("gpuMatrix Single Precision Euclidean Distance",
{
    
    has_cpu_skip()
    
    fgpuX <- gpuMatrix(A, type="float")
    
    E <- dist(fgpuX)
    
    expect_equal(E[], D, tolerance=1e-06, 
                 info="float euclidean distances not equivalent",
                 check.attributes=FALSE)  
})

test_that("gpuMatrix Double Precision Euclidean Distance", 
{
    
    has_cpu_skip()
    
    fgpuX <- gpuMatrix(A, type="double")
    
    E <- dist(fgpuX)
    
    expect_equal(E[], D, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double euclidean distances not equivalent",
                 check.attributes=FALSE) 
})

options(gpuR.default.device.type = "gpu")
