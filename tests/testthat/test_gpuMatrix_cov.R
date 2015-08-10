library(gpuR)
context("gpuMatrix Correlations")

# set seed
set.seed(123)

ORDER_X <- 4
ORDER_Y <- 5

# Base R objects
A <- matrix(rnorm(ORDER_X*ORDER_Y), nrow=ORDER_X, ncol=ORDER_Y)

C <- cov(A)


test_that("gpuMatrix Single Precision Pearson Covariance",
{
    has_gpu_skip()
    
    fgpuX <- gpuMatrix(A, type="float")
    
    gpuC <- cov(fgpuX)
    
    expect_is(gpuC, "fgpuMatrix")
    expect_equal(gpuC[], C, tolerance=1e-06, 
                 info="float covariance values not equivalent")  
})

test_that("gpuMatrix Double Precision Pearson Covariance", 
{
    
    has_gpu_skip()
    has_double_skip()
    
    dgpuX <- gpuMatrix(A, type="double")
    
    gpuC <- cov(dgpuX)
    
    expect_is(gpuC, "dgpuMatrix")
    expect_equal(gpuC[], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double colSums not equivalent")  
})
