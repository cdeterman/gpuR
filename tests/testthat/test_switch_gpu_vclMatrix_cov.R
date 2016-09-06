library(gpuR)
context("Switching GPUs vclMatrix Correlations")

# set seed
set.seed(123)

ORDER_X <- 4
ORDER_Y <- 5

# Base R objects
A <- matrix(rnorm(ORDER_X*ORDER_Y), nrow=ORDER_X, ncol=ORDER_Y)

C <- cov(A)


test_that("Switching GPUs vclMatrix Single Precision Pearson Covariance",
{
    has_multiple_gpu_skip()
    
    setContext(2L)
    
    fgpuX <- vclMatrix(A, type="float")
    
    setContext(1L)
    
    gpuC <- cov(fgpuX)
    
    expect_is(gpuC, "fvclMatrix")
    expect_equal(gpuC[], C, tolerance=1e-06, 
                 info="float covariance values not equivalent")  
    expect_equal(gpuC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPUs vclMatrix Double Precision Pearson Covariance", 
{
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    setContext(2L)
    
    dgpuX <- vclMatrix(A, type="double")
    
    setContext(1L)
    
    gpuC <- cov(dgpuX)
    
    expect_is(gpuC, "dvclMatrix")
    expect_equal(gpuC[], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double covariance values not equivalent")  
    expect_equal(gpuC@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})
