library(gpuR)
context("CPU vclMatrix Distance Computations")

# set option to use CPU instead of GPU
options(gpuR.default.device.type = "cpu")

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)

D <- as.matrix(dist(A))

test_that("vclMatrix Single Precision Euclidean Distance",
{
    
    has_cpu_skip()
    
    fgpuX <- vclMatrix(A, type="float")
    
    E <- dist(fgpuX)
    
    expect_equal(E[], D, tolerance=1e-06, 
                 info="float euclidean distances not equivalent",
                 check.attributes=FALSE)  
})

test_that("vclMatrix Double Precision Euclidean Distance", 
{
    
    has_cpu_skip()
    
    fgpuX <- vclMatrix(A, type="double")
    
    E <- dist(fgpuX)
    
    expect_equal(E[], D, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double euclidean distances not equivalent",
                 check.attributes=FALSE) 
})

options(gpuR.default.device.type = "gpu")

