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
sqD <- D^2

test_that("CPU vclMatrixSingle Precision Euclidean Distance",
{
    
    has_cpu_skip()
    
    fgpuX <- vclMatrix(A, type="float")
    
    E <- dist(fgpuX)
    
    expect_equal(E[], D, tolerance=1e-06, 
                 info="float euclidean distances not equivalent",
                 check.attributes=FALSE)  
})

test_that("CPU vclMatrixDouble Precision Euclidean Distance", 
{
    
    has_cpu_skip()
    
    fgpuX <- vclMatrix(A, type="double")
    
    E <- dist(fgpuX)
    
    expect_equal(E[], D, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double euclidean distances not equivalent",
                 check.attributes=FALSE) 
})

test_that("CPU vclMatrixSingle Precision Squared Euclidean Distance",
{
    has_cpu_skip()
    
    fgpuX <- vclMatrix(A, type="float")
    
    E <- dist(fgpuX, method = "sqEuclidean")
    
    expect_equal(E[], sqD, tolerance=1e-06, 
                 info="float squared euclidean distances not equivalent",
                 check.attributes=FALSE)  
})

test_that("CPU vclMatrixDouble Precision Squared Euclidean Distance", 
{
    has_cpu_skip()
    
    fgpuX <- vclMatrix(A, type="double")
    
    E <- dist(fgpuX, method = "sqEuclidean")
    
    expect_equal(E[], sqD, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double squared euclidean distances not equivalent",
                 check.attributes=FALSE) 
})

options(gpuR.default.device.type = "gpu")

