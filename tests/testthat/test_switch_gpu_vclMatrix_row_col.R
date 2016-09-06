library(gpuR)
context("Switching GPU vclMatrix Row and Column Methods")

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

RS <- rowSums(A[2:4, 2:4])
CS <- colSums(A[2:4, 2:4])
RMS <- rowMeans(A[2:4, 2:4])
CMS <- colMeans(A[2:4, 2:4])


test_that("Switching GPU vclMatrix Single Precision Column Sums",
{
    
    has_multiple_gpu_skip()
    
    setContext(2L)
    
    fgpuX <- vclMatrix(A, type="float")
    
    setContext(1L)
    
    gpuC <- colSums(fgpuX)
    
    expect_is(gpuC, "fvclVector")
    expect_equal(gpuC[], C, tolerance=1e-06, 
                 info="float covariance values not equivalent")  
    expect_equal(gpuC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Double Precision Column Sums", 
{
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    setContext(2L)
    
    dgpuX <- vclMatrix(A, type="double")
    
    setContext(1L)
    
    gpuC <- colSums(dgpuX)
    
    expect_is(gpuC, "dvclVector")
    expect_equal(gpuC[], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double colSums not equivalent")  
    expect_equal(gpuC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})


test_that("Switching GPU vclMatrix Single Precision Row Sums",
{
    
    has_multiple_gpu_skip()
    
    setContext(2L)
    
    fgpuX <- vclMatrix(A, type="float")
    
    setContext(1L)
    
    gpuC <- rowSums(fgpuX)
    
    expect_is(gpuC, "fvclVector")
    expect_equal(gpuC[], R, tolerance=1e-06, 
                 info="float covariance values not equivalent")  
    expect_equal(gpuC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Double Precision Row Sums", 
{
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    setContext(2L)
    
    dgpuX <- vclMatrix(A, type="double")
    
    setContext(1L)
    
    gpuC <- rowSums(dgpuX)
    
    expect_is(gpuC, "dvclVector")
    expect_equal(gpuC[], R, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double colSums not equivalent")  
    expect_equal(gpuC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Single Precision Column Means",
{
    
    has_multiple_gpu_skip()
    
    setContext(2L)
    
    fgpuX <- vclMatrix(A, type="float")
    
    setContext(1L)
    
    gpuC <- colMeans(fgpuX)
    
    expect_is(gpuC, "fvclVector")
    expect_equal(gpuC[], CM, tolerance=1e-06, 
                 info="float covariance values not equivalent")  
    expect_equal(gpuC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Double Precision Column Means", 
{
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    setContext(2L)
    
    dgpuX <- vclMatrix(A, type="double")
    
    setContext(1L)
    
    gpuC <- colMeans(dgpuX)
    
    expect_is(gpuC, "dvclVector")
    expect_equal(gpuC[], CM, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double colSums not equivalent")  
    expect_equal(gpuC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})


test_that("Switching GPU vclMatrix Single Precision Row Means",
{
    
    has_multiple_gpu_skip()
    
    setContext(2L)
    
    fgpuX <- vclMatrix(A, type="float")
    
    setContext(1L)
    
    gpuC <- rowMeans(fgpuX)
    
    expect_is(gpuC, "fvclVector")
    expect_equal(gpuC[], RM, tolerance=1e-06, 
                 info="float covariance values not equivalent")  
    expect_equal(gpuC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Double Precision Row Means", 
{
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    setContext(2L)
    
    dgpuX <- vclMatrix(A, type="double")
    
    setContext(1L)
    
    gpuC <- rowMeans(dgpuX)
    
    expect_is(gpuC, "dvclVector")
    expect_equal(gpuC[], RM, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double colSums not equivalent")  
    expect_equal(gpuC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

#cbind/rbind tests
test_that("Switching GPU vclMatrix Single Precision cbind",
{
    has_multiple_gpu_skip()
    
    C_bind <- cbind(A, B)
    C_scalar <- cbind(1, A)
    C_scalar2 <- cbind(A,1)
    
    setContext(2L)
    
    gpuA <- vclMatrix(A, type="float")
    gpuB <- vclMatrix(B, type="float")
    
    setContext(1L)
    
    gpuC <- cbind(gpuA, gpuB)
    
    expect_is(gpuC, "fvclMatrix")
    expect_equal(gpuC[], C_bind, tolerance=1e-06, 
                 info="float cbind not equivalent")  
    expect_equal(gpuC@.context_index, 2L,
                 info = "context index not assigned properly")
    
    gpu_scalar <- cbind(1, gpuA)
    gpu_scalar2 <- cbind(gpuA, 1)
    
    expect_equal(gpu_scalar[], C_scalar, tolerance=1e-06, 
                 info="float scalar cbind not equivalent") 
    expect_equal(gpu_scalar2[], C_scalar2, tolerance=1e-06, 
                 info="float scalar cbind not equivalent") 
    expect_equal(gpu_scalar@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(gpu_scalar2@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Double Precision cbind",
{
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    C_bind <- cbind(A, B)
    C_scalar <- cbind(1, A)
    C_scalar2 <- cbind(A,1)
    
    setContext(2L)
    
    gpuA <- vclMatrix(A, type="double")
    gpuB <- vclMatrix(B, type="double")
    
    setContext(1L)
    
    gpuC <- cbind(gpuA, gpuB)
    
    expect_is(gpuC, "dvclMatrix")
    expect_equal(gpuC[], C_bind, tolerance=.Machine$double.eps^0.5, 
                 info="double cbind not equivalent")  
    expect_equal(gpuC@.context_index, 2L,
                 info = "context index not assigned properly")
    
    gpu_scalar <- cbind(1, gpuA)
    gpu_scalar2 <- cbind(gpuA, 1)
    
    expect_equal(gpu_scalar[], C_scalar, tolerance=.Machine$double.eps^0.5, 
                 info="double scalar cbind not equivalent") 
    expect_equal(gpu_scalar2[], C_scalar2, tolerance=.Machine$double.eps^0.5, 
                 info="double scalar cbind not equivalent") 
    expect_equal(gpu_scalar@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(gpu_scalar2@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Single Precision rbind",
{
    has_multiple_gpu_skip()
    
    C_bind <- rbind(A, B)
    C_scalar <- rbind(1, A)
    C_scalar2 <- rbind(A,1)
    
    setContext(2L)
    
    gpuA <- vclMatrix(A, type="float")
    gpuB <- vclMatrix(B, type="float")
    
    setContext(1L)
    
    gpuC <- rbind(gpuA, gpuB)
    
    expect_is(gpuC, "fvclMatrix")
    expect_equal(gpuC[], C_bind, tolerance=1e-06, 
                 info="float rbind not equivalent")  
    expect_equal(gpuC@.context_index, 2L,
                 info = "context index not assigned properly")
    
    gpu_scalar <- rbind(1, gpuA)
    gpu_scalar2 <- rbind(gpuA, 1)
    
    expect_equal(gpu_scalar[], C_scalar, tolerance=1e-06, 
                 info="float scalar rbind not equivalent") 
    expect_equal(gpu_scalar2[], C_scalar2, tolerance=1e-06, 
                 info="float scalar rbind not equivalent") 
    expect_equal(gpu_scalar@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(gpu_scalar2@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Double Precision rbind",
{
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    C_bind <- rbind(A, B)
    C_scalar <- rbind(1, A)
    C_scalar2 <- rbind(A,1)
    
    setContext(2L)
    
    gpuA <- vclMatrix(A, type="double")
    gpuB <- vclMatrix(B, type="double")
    
    setContext(1L)
    
    gpuC <- rbind(gpuA, gpuB)
    
    expect_is(gpuC, "dvclMatrix")
    expect_equal(gpuC[], C_bind, tolerance=.Machine$double.eps^0.5, 
                 info="double rbind not equivalent")  
    expect_equal(gpuC@.context_index, 2L,
                 info = "context index not assigned properly")
    
    gpu_scalar <- rbind(1, gpuA)
    gpu_scalar2 <- rbind(gpuA, 1)
    
    expect_equal(gpu_scalar[], C_scalar, tolerance=.Machine$double.eps^0.5, 
                 info="double scalar rbind not equivalent") 
    expect_equal(gpu_scalar2[], C_scalar2, tolerance=.Machine$double.eps^0.5, 
                 info="double scalar rbind not equivalent") 
    expect_equal(gpu_scalar@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(gpu_scalar2@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

# 'block' object tests

test_that("Switching GPU vclMatrix Single Precision Block Column Sums",
{
    
    has_multiple_gpu_skip()
    
    setContext(2L)
    
    fgpuX <- vclMatrix(A, type="float")
    fgpuXS <- block(fgpuX, 2L,4L,2L,4L)
    
    setContext(1L)
    
    gpuC <- colSums(fgpuXS)
    
    expect_is(gpuC, "fvclVector")
    expect_equal(gpuC[], CS, tolerance=1e-06, 
                 info="float colSums not equivalent") 
    expect_equal(gpuC@.context_index, 2L,
                 info = "context index not assigned properly") 
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Double Precision Block Column Sums", 
{
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    setContext(2L)
    
    dgpuX <- vclMatrix(A, type="double")
    dgpuXS <- block(dgpuX, 2L,4L,2L,4L)
    
    setContext(1L)
    
    gpuC <- colSums(dgpuXS)
    
    expect_is(gpuC, "dvclVector")
    expect_equal(gpuC[], CS, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double colSums not equivalent")  
    expect_equal(gpuC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})


test_that("Switching GPU vclMatrix Single Precision Block Row Sums",
{
    
    has_multiple_gpu_skip()
    
    setContext(2L)
    
    fgpuX <- vclMatrix(A, type="float")
    fgpuXS <- block(fgpuX, 2L,4L,2L,4L)
    
    setContext(1L)
    
    gpuC <- rowSums(fgpuXS)
    
    expect_is(gpuC, "fvclVector")
    expect_equal(gpuC[], RS, tolerance=1e-06, 
                 info="float rowSums not equivalent")  
    expect_equal(gpuC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Double Precision Block Row Sums", 
{
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    setContext(2L)
    
    dgpuX <- vclMatrix(A, type="double")
    dgpuXS <- block(dgpuX, 2L,4L,2L,4L)
    
    setContext(1L)
    
    gpuC <- rowSums(dgpuXS)
    
    expect_is(gpuC, "dvclVector")
    expect_equal(gpuC[], RS, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double colSums not equivalent") 
    expect_equal(gpuC@.context_index, 2L,
                 info = "context index not assigned properly") 
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Single Precision Block Column Means",
{
    
    has_multiple_gpu_skip()
    
    setContext(2L)
    
    fgpuX <- vclMatrix(A, type="float")
    fgpuXS <- block(fgpuX, 2L,4L,2L,4L)
    
    setContext(1L)
    
    gpuC <- colMeans(fgpuXS)
    
    expect_is(gpuC, "fvclVector")
    expect_equal(gpuC[], CMS, tolerance=1e-06, 
                 info="float colMeans not equivalent")  
    expect_equal(gpuC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Double Precision Block Column Means", 
{
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    setContext(2L)
    
    dgpuX <- vclMatrix(A, type="double")
    dgpuXS <- block(dgpuX, 2L,4L,2L,4L)
    
    setContext(1L)
    
    gpuC <- colMeans(dgpuXS)
    
    expect_is(gpuC, "dvclVector")
    expect_equal(gpuC[], CMS, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double colMeans not equivalent")  
    expect_equal(gpuC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})


test_that("Switching GPU vclMatrix Single Precision Block Row Means",
{
    has_multiple_gpu_skip()
    
    setContext(2L)
    
    fgpuX <- vclMatrix(A, type="float")
    fgpuXS <- block(fgpuX, 2L,4L,2L,4L)
    
    setContext(1L)
    
    gpuC <- rowMeans(fgpuXS)
    
    expect_is(gpuC, "fvclVector")
    expect_equal(gpuC[], RMS, tolerance=1e-06, 
                 info="float rowMeans not equivalent")  
    expect_equal(gpuC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Double Precision Block Row Means", 
{
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    setContext(2L)
    
    dgpuX <- vclMatrix(A, type="double")
    dgpuXS <- block(dgpuX, 2L,4L,2L,4L)
    
    setContext(1L)
    
    gpuC <- rowMeans(dgpuXS)
    
    expect_is(gpuC, "dvclVector")
    expect_equal(gpuC[], RMS, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double rowMeans not equivalent")  
    expect_equal(gpuC@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})
