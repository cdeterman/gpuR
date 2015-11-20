library(gpuR)
context("CPU gpuMatrix Row and Column Methods")

# set option to use CPU instead of GPU
options(gpuR.default.device = "cpu")

# set seed
set.seed(123)

ORDER_X <- 4
ORDER_Y <- 5

# Base R objects
A <- matrix(rnorm(ORDER_X*ORDER_Y), nrow=ORDER_X, ncol=ORDER_Y)

R <- rowSums(A)
C <- colSums(A)
RM <- rowMeans(A)
CM <- colMeans(A)


test_that("CPU gpuMatrix Single Precision Column Sums",
{
    has_cpu_skip()
    
    fgpuX <- gpuMatrix(A, type="float")
    
    gpuC <- colSums(fgpuX)
    
    expect_is(gpuC, "fgpuVector")
    expect_equal(gpuC[], C, tolerance=1e-06, 
                 info="float covariance values not equivalent")  
})

test_that("CPU gpuMatrix Double Precision Column Sums", 
{
    has_cpu_skip()
    
    dgpuX <- gpuMatrix(A, type="double")
    
    gpuC <- colSums(dgpuX)
    
    expect_is(gpuC, "dgpuVector")
    expect_equal(gpuC[], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double colSums not equivalent")  
})

test_that("CPU gpuMatrix Single Precision Row Sums",
{
    has_cpu_skip()
    
    fgpuX <- gpuMatrix(A, type="float")
    
    gpuC <- rowSums(fgpuX)
    
    expect_is(gpuC, "fgpuVector")
    expect_equal(gpuC[], R, tolerance=1e-06, 
                 info="float covariance values not equivalent")  
})

test_that("CPU gpuMatrix Double Precision Row Sums", 
{
    has_cpu_skip()
    
    dgpuX <- gpuMatrix(A, type="double")
    
    gpuC <- rowSums(dgpuX)
    
    expect_is(gpuC, "dgpuVector")
    expect_equal(gpuC[], R, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double colSums not equivalent")  
})

test_that("CPU gpuMatrix Single Precision Column Means",
{
    has_cpu_skip()
    
    fgpuX <- gpuMatrix(A, type="float")
    
    gpuC <- colMeans(fgpuX)
    
    expect_is(gpuC, "fgpuVector")
    expect_equal(gpuC[], CM, tolerance=1e-06, 
                 info="float covariance values not equivalent")  
})

test_that("CPU gpuMatrix Double Precision Column Means", 
{
    has_cpu_skip()
    
    dgpuX <- gpuMatrix(A, type="double")
    
    gpuC <- colMeans(dgpuX)
    
    expect_is(gpuC, "dgpuVector")
    expect_equal(gpuC[], CM, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double colSums not equivalent")  
})


test_that("CPU gpuMatrix Single Precision Row Means",
{
    has_cpu_skip()
    fgpuX <- gpuMatrix(A, type="float")
    
    gpuC <- rowMeans(fgpuX)
    
    expect_is(gpuC, "fgpuVector")
    expect_equal(gpuC[], RM, tolerance=1e-06, 
                 info="float covariance values not equivalent")  
})

test_that("CPU gpuMatrix Double Precision Row Means", 
{
    has_cpu_skip()
    dgpuX <- gpuMatrix(A, type="double")
    
    gpuC <- rowMeans(dgpuX)
    
    expect_is(gpuC, "dgpuVector")
    expect_equal(gpuC[], RM, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double colSums not equivalent")  
})

options(gpuR.default.device = "gpu")
