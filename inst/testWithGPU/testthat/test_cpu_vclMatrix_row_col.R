library(gpuR)
context("CPU vclMatrix Row and Column Methods")

# set option to use CPU instead of GPU
options(gpuR.default.device.type = "cpu")

# set seed
set.seed(123)

ORDER_X <- 4
ORDER_Y <- 5

# Base R objects
A <- matrix(rnorm(ORDER_X*ORDER_Y), nrow=ORDER_X, ncol=ORDER_Y)
B <- matrix(rnorm(ORDER_X*ORDER_Y), nrow=ORDER_X, ncol=ORDER_Y)

R <- rowSums(A)
C <- colSums(A)
RM <- rowMeans(A)
CM <- colMeans(A)


test_that("CPU vclMatrix Single Precision Column Sums",
{
    has_cpu_skip()
    
    fgpuX <- vclMatrix(A, type="float")
    
    gpuC <- colSums(fgpuX)
    
    expect_is(gpuC, "fvclVector")
    expect_equal(gpuC[], C, tolerance=1e-06, 
                 info="float covariance values not equivalent")  
})

test_that("CPU vclMatrix Double Precision Column Sums", 
{
    has_cpu_skip()
    
    dgpuX <- vclMatrix(A, type="double")
    
    gpuC <- colSums(dgpuX)
    
    expect_is(gpuC, "dvclVector")
    expect_equal(gpuC[], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double colSums not equivalent")  
})

test_that("CPU vclMatrix Single Precision Row Sums",
{
    has_cpu_skip()
    
    fgpuX <- vclMatrix(A, type="float")
    
    gpuC <- rowSums(fgpuX)
    
    expect_is(gpuC, "fvclVector")
    expect_equal(gpuC[], R, tolerance=1e-06, 
                 info="float covariance values not equivalent")  
})

test_that("CPU vclMatrix Double Precision Row Sums", 
{
    has_cpu_skip()
    
    dgpuX <- vclMatrix(A, type="double")
    
    gpuC <- rowSums(dgpuX)
    
    expect_is(gpuC, "dvclVector")
    expect_equal(gpuC[], R, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double colSums not equivalent")  
})

test_that("CPU vclMatrix Single Precision Column Means",
{
    has_cpu_skip()
    
    fgpuX <- vclMatrix(A, type="float")
    
    gpuC <- colMeans(fgpuX)
    
    expect_is(gpuC, "fvclVector")
    expect_equal(gpuC[], CM, tolerance=1e-06, 
                 info="float covariance values not equivalent")  
})

test_that("CPU vclMatrix Double Precision Column Means", 
{
    has_cpu_skip()
    
    dgpuX <- vclMatrix(A, type="double")
    
    gpuC <- colMeans(dgpuX)
    
    expect_is(gpuC, "dvclVector")
    expect_equal(gpuC[], CM, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double colSums not equivalent")  
})


test_that("CPU vclMatrix Single Precision Row Means",
{
    has_cpu_skip()
    
    fgpuX <- vclMatrix(A, type="float")
    
    gpuC <- rowMeans(fgpuX)
    
    expect_is(gpuC, "fvclVector")
    expect_equal(gpuC[], RM, tolerance=1e-06, 
                 info="float covariance values not equivalent")  
})

test_that("CPU vclMatrix Double Precision Row Means", 
{
    has_cpu_skip()
    
    dgpuX <- vclMatrix(A, type="double")
    
    gpuC <- rowMeans(dgpuX)
    
    expect_is(gpuC, "dvclVector")
    expect_equal(gpuC[], RM, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double colSums not equivalent")  
})

#cbind/rbind tests
test_that("CPU vclMatrix Single Precision cbind",
{
    has_cpu_skip()
    
    C_bind <- cbind(A, B)
    C_scalar <- cbind(1, A)
    C_scalar2 <- cbind(A,1)
    
    gpuA <- vclMatrix(A, type="float")
    gpuB <- vclMatrix(B, type="float")
    
    gpuC <- cbind(gpuA, gpuB)
    
    expect_is(gpuC, "fvclMatrix")
    expect_equal(gpuC[], C_bind, tolerance=1e-06, 
                 info="float cbind not equivalent")  
    
    gpu_scalar <- cbind(1, gpuA)
    gpu_scalar2 <- cbind(gpuA, 1)
    
    expect_equal(gpu_scalar[], C_scalar, tolerance=1e-06, 
                 info="float scalar cbind not equivalent") 
    expect_equal(gpu_scalar2[], C_scalar2, tolerance=1e-06, 
                 info="float scalar cbind not equivalent") 
})

test_that("CPU vclMatrix Double Precision cbind",
{
    has_cpu_skip()
    
    C_bind <- cbind(A, B)
    C_scalar <- cbind(1, A)
    C_scalar2 <- cbind(A,1)
    
    gpuA <- vclMatrix(A, type="double")
    gpuB <- vclMatrix(B, type="double")
    
    gpuC <- cbind(gpuA, gpuB)
    
    expect_is(gpuC, "dvclMatrix")
    expect_equal(gpuC[], C_bind, tolerance=.Machine$double.eps^0.5, 
                 info="double cbind not equivalent")  
    
    gpu_scalar <- cbind(1, gpuA)
    gpu_scalar2 <- cbind(gpuA, 1)
    
    expect_equal(gpu_scalar[], C_scalar, tolerance=.Machine$double.eps^0.5, 
                 info="double scalar cbind not equivalent") 
    expect_equal(gpu_scalar2[], C_scalar2, tolerance=.Machine$double.eps^0.5, 
                 info="double scalar cbind not equivalent") 
})

test_that("CPU vclMatrix Single Precision rbind",
{
    has_cpu_skip()
    
    C_bind <- rbind(A, B)
    C_scalar <- rbind(1, A)
    C_scalar2 <- rbind(A,1)
    
    gpuA <- vclMatrix(A, type="float")
    gpuB <- vclMatrix(B, type="float")
    
    gpuC <- rbind(gpuA, gpuB)
    
    expect_is(gpuC, "fvclMatrix")
    expect_equal(gpuC[], C_bind, tolerance=1e-06, 
                 info="float rbind not equivalent")  
    
    gpu_scalar <- rbind(1, gpuA)
    gpu_scalar2 <- rbind(gpuA, 1)
    
    expect_equal(gpu_scalar[], C_scalar, tolerance=1e-06, 
                 info="float scalar rbind not equivalent") 
    expect_equal(gpu_scalar2[], C_scalar2, tolerance=1e-06, 
                 info="float scalar rbind not equivalent") 
})

test_that("CPU vclMatrix Double Precision rbind",
{
    has_cpu_skip()
    
    C_bind <- rbind(A, B)
    C_scalar <- rbind(1, A)
    C_scalar2 <- rbind(A,1)
    
    gpuA <- vclMatrix(A, type="double")
    gpuB <- vclMatrix(B, type="double")
    
    gpuC <- rbind(gpuA, gpuB)
    
    expect_is(gpuC, "dvclMatrix")
    expect_equal(gpuC[], C_bind, tolerance=.Machine$double.eps^0.5, 
                 info="double rbind not equivalent")  
    
    gpu_scalar <- rbind(1, gpuA)
    gpu_scalar2 <- rbind(gpuA, 1)
    
    expect_equal(gpu_scalar[], C_scalar, tolerance=.Machine$double.eps^0.5, 
                 info="double scalar rbind not equivalent") 
    expect_equal(gpu_scalar2[], C_scalar2, tolerance=.Machine$double.eps^0.5, 
                 info="double scalar rbind not equivalent") 
})

options(gpuR.default.device.type = "gpu")
