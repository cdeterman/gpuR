library(gpuR)
context("vclVector shared memory")

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)

test_that("Share memory between vclMatrix & vclVector", {
    
    has_gpu_skip()
    
    gpuA <- vclMatrix(A, type = "float")
    
    # convert to vector
    gpuB <- as.vclVector(gpuA, shared = TRUE)
    
    expect_is(gpuB, "fvclVector")
    expect_equal(gpuA[1], gpuB[1], tolerance=1e-07, 
                 info="float elements not equivalent")
    
    has_double_skip()
    
    gpuA <- vclMatrix(A)
    
    # convert to vector
    gpuB <- as.vclVector(gpuA, shared = TRUE)
    
    gpuB[1] <- 42
    
    expect_is(gpuB, "dvclVector")
    expect_equal(gpuA[1], gpuB[1], tolerance=.Machine$double.eps^0.5, 
                 info="double elements not equivalent")
})

