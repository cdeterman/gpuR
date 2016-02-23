library(gpuR)
context("vclMatrix Distance Computations")

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
B <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)

D <- as.matrix(dist(A))
sqD <- D^2

# (A^2 %*% matrix(1, nrow=ncol(A))) %*% t(matrix(1, nrow=nrow(B)))
# matrix(1, nrow=nrow(A)) %*% t(B^2 %*% matrix(1, nrow=ncol(B)))

pD <- matrix(0, nrow=nrow(A), ncol=nrow(B))
# Pairwise check
for(i in 1:nrow(A)){
    for(j in 1:nrow(B)){
        pD[i,j] <- sqrt(sum((A[i,] - B[j,])^2))
    }
}

sqpD <- pD^2

test_that("vclMatrix Single Precision Euclidean Distance",
{
    
    has_gpu_skip()
    
    fgpuX <- vclMatrix(A, type="float")
    
    E <- dist(fgpuX)
    
    expect_equal(E[], D, tolerance=1e-06, 
                 info="float euclidean distances not equivalent",
                 check.attributes=FALSE)  
})

test_that("vclMatrix Double Precision Euclidean Distance", 
{
    
    has_gpu_skip()
    has_double_skip()
    
    fgpuX <- vclMatrix(A, type="double")
    
    E <- dist(fgpuX)
    
    expect_equal(E[], D, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double euclidean distances not equivalent",
                 check.attributes=FALSE) 
})

test_that("vclMatrix Single Precision Squared Euclidean Distance",
{
    
    has_gpu_skip()
    
    fgpuX <- vclMatrix(A, type="float")
    
    E <- dist(fgpuX, method = "sqEuclidean")
    
    expect_equal(E[], sqD, tolerance=1e-06, 
                 info="float squared euclidean distances not equivalent",
                 check.attributes=FALSE)  
})

test_that("vclMatrix Double Precision Squared Euclidean Distance", 
{
    
    has_gpu_skip()
    has_double_skip()
    
    fgpuX <- vclMatrix(A, type="double")
    
    E <- dist(fgpuX, method = "sqEuclidean")
    
    expect_equal(E[], sqD, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double squared euclidean distances not equivalent",
                 check.attributes=FALSE) 
})

test_that("vclMatrix Single Precision Pairwise Euclidean Distance",
{
    
    has_gpu_skip()
    
    fgpuA <- vclMatrix(A, type="float")
    fgpuB <- vclMatrix(B, type="float")
    
    E <- distance(fgpuA, fgpuB)
    
    expect_equal(E[], pD, tolerance=1e-06, 
                 info="float euclidean pairwise distances not equivalent",
                 check.attributes=FALSE)  
})

test_that("vclMatrix Double Precision Pairwise Euclidean Distance", 
{
    
    has_gpu_skip()
    has_double_skip()
    
    fgpuA <- vclMatrix(A, type="double")
    fgpuB <- vclMatrix(B, type="double")
    
    E <- distance(fgpuA, fgpuB)
    
    expect_equal(E[], pD, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double euclidean pairwise distances not equivalent",
                 check.attributes=FALSE) 
})

test_that("vclMatrix Single Precision Pairwise Squared Euclidean Distance",
{
    
    has_gpu_skip()
    
    fgpuA <- vclMatrix(A, type="float")
    fgpuB <- vclMatrix(B, type="float")
    
    E <- distance(fgpuA, fgpuB, method = "sqEuclidean")
    
    expect_equal(E[], sqpD, tolerance=1e-06, 
                 info="float squared euclidean pairwise distances not equivalent",
                 check.attributes=FALSE)  
})

test_that("vclMatrix Double Precision Pairwise Squared Euclidean Distance", 
{
    
    has_gpu_skip()
    has_double_skip()
    
    fgpuA <- vclMatrix(A, type="double")
    fgpuB <- vclMatrix(B, type="double")
    
    E <- distance(fgpuA, fgpuB, method = "sqEuclidean")
    
    expect_equal(E[], sqpD, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double squared euclidean pairwise distances not equivalent",
                 check.attributes=FALSE) 
})


