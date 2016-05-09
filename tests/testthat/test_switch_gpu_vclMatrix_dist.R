library(gpuR)
context("Switching GPU vclMatrix Distance Computations")

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
B <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)

# none square matrix
C <- matrix(rnorm(20), nrow = 5, ncol = 4)

# matrix for error check
G <- matrix(rnorm(25), nrow = 5, ncol = 5)

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

lpD <- matrix(0, nrow=nrow(A), ncol=nrow(C))
# Pairwise check
for(i in 1:nrow(A)){
    for(j in 1:nrow(C)){
        lpD[i,j] <- sqrt(sum((A[i,] - C[j,])^2))
    }
}

rpD <- matrix(0, nrow=nrow(C), ncol=nrow(A))
# Pairwise check
for(i in 1:nrow(C)){
    for(j in 1:nrow(A)){
        rpD[i,j] <- sqrt(sum((C[i,] - A[j,])^2))
    }
}

sqpD <- pD^2
lsqpD <- lpD^2
rsqpD <- rpD^2

test_that("Switching GPU vclMatrix Single Precision Euclidean Distance",
{
    
    has_multiple_gpu_skip()
    
    setContext(2L)
    
    fgpuX <- vclMatrix(A, type="float")
    
    setContext(1L)
    
    E <- dist(fgpuX)
    
    expect_equal(E[], D, tolerance=1e-06, 
                 info="float euclidean distances not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuX@.context_index, 2L,
                 info = "context index not assigned properly")
    expect_equal(E@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Double Precision Euclidean Distance", 
{
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    setContext(2L)
    
    fgpuX <- vclMatrix(A, type="double")
    
    setContext(1L)
    
    E <- dist(fgpuX)
    
    expect_equal(E[], D, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double euclidean distances not equivalent",
                 check.attributes=FALSE) 
    expect_equal(E@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Single Precision Squared Euclidean Distance",
{
    
    has_multiple_gpu_skip()
    
    setContext(2L)
    
    fgpuX <- vclMatrix(A, type="float")
    
    setContext(1L)
    
    E <- dist(fgpuX, method = "sqEuclidean")
    
    expect_equal(E[], sqD, tolerance=1e-06, 
                 info="float squared euclidean distances not equivalent",
                 check.attributes=FALSE)  
    expect_equal(E@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Double Precision Squared Euclidean Distance", 
{
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    setContext(2L)
    
    fgpuX <- vclMatrix(A, type="double")
    
    setContext(1L)
    
    E <- dist(fgpuX, method = "sqEuclidean")
    
    expect_equal(E[], sqD, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double squared euclidean distances not equivalent",
                 check.attributes=FALSE) 
    expect_equal(E@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Single Precision Pairwise Euclidean Distance",
{
    
    has_multiple_gpu_skip()
    
    setContext(2L)
    
    fgpuA <- vclMatrix(A, type="float")
    fgpuB <- vclMatrix(B, type="float")
    fgpuC <- vclMatrix(C, type="float")
    
    setContext(1L)
    
    E <- distance(fgpuA, fgpuB)
    lE <- distance(fgpuA, fgpuC)
    rE <- distance(fgpuC, fgpuA)
    
    expect_equal(E[], pD, tolerance=1e-06, 
                 info="float euclidean pairwise distances not equivalent",
                 check.attributes=FALSE)  
    expect_equal(lE[], lpD, tolerance=1e-06, 
                 info="float euclidean pairwise distances not equivalent",
                 check.attributes=FALSE)  
    expect_equal(rE[], rpD, tolerance=1e-06, 
                 info="float euclidean pairwise distances not equivalent",
                 check.attributes=FALSE)  
    expect_equal(E@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(lE@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(rE@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Double Precision Pairwise Euclidean Distance", 
{
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    setContext(2L)
    
    fgpuA <- vclMatrix(A, type="double")
    fgpuB <- vclMatrix(B, type="double")
    fgpuC <- vclMatrix(C, type="double")
    
    setContext(1L)
    
    E <- distance(fgpuA, fgpuB)
    lE <- distance(fgpuA, fgpuC)
    rE <- distance(fgpuC, fgpuA)
    
    expect_equal(E[], pD, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double euclidean pairwise distances not equivalent",
                 check.attributes=FALSE) 
    expect_equal(lE[], lpD, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double euclidean pairwise distances not equivalent",
                 check.attributes=FALSE) 
    expect_equal(rE[], rpD, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double euclidean pairwise distances not equivalent",
                 check.attributes=FALSE) 
    expect_equal(E@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(lE@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(rE@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Single Precision Pairwise Squared Euclidean Distance",
{
    
    has_multiple_gpu_skip()
    
    setContext(2L)
    
    fgpuA <- vclMatrix(A, type="float")
    fgpuB <- vclMatrix(B, type="float")
    fgpuC <- vclMatrix(C, type="float")
    
    setContext(1L)
    
    E <- distance(fgpuA, fgpuB, method = "sqEuclidean")
    lE <- distance(fgpuA, fgpuC, method = "sqEuclidean")
    rE <- distance(fgpuC, fgpuA, method = "sqEuclidean")
    
    expect_equal(E[], sqpD, tolerance=1e-06, 
                 info="float squared euclidean pairwise distances not equivalent",
                 check.attributes=FALSE) 
    expect_equal(lE[], lsqpD, tolerance=1e-06, 
                 info="float squared euclidean pairwise distances not equivalent",
                 check.attributes=FALSE) 
    expect_equal(rE[], rsqpD, tolerance=1e-06, 
                 info="float squared euclidean pairwise distances not equivalent",
                 check.attributes=FALSE) 
    expect_equal(E@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(lE@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(rE@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

test_that("Switching GPU vclMatrix Double Precision Pairwise Squared Euclidean Distance", 
{
    
    has_multiple_gpu_skip()
    has_multiple_double_skip()
    
    setContext(2L)
    
    fgpuA <- vclMatrix(A, type="double")
    fgpuB <- vclMatrix(B, type="double")
    fgpuC <- vclMatrix(C, type="double")
    
    setContext(1L)
    
    E <- distance(fgpuA, fgpuB, method = "sqEuclidean")
    lE <- distance(fgpuA, fgpuC, method = "sqEuclidean")
    rE <- distance(fgpuC, fgpuA, method = "sqEuclidean")
    
    expect_equal(E[], sqpD, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double squared euclidean pairwise distances not equivalent",
                 check.attributes=FALSE) 
    expect_equal(lE[], lsqpD, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double squared euclidean pairwise distances not equivalent",
                 check.attributes=FALSE) 
    expect_equal(rE[], rsqpD, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double squared euclidean pairwise distances not equivalent",
                 check.attributes=FALSE) 
    expect_equal(E@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(lE@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(rE@.context_index, 2L, 
                 info = "context index hasn't been assigned correctly")
    expect_equal(currentContext(), 1L, 
                 info = "context index has been change unintentionally")
})

