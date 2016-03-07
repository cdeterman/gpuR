library(gpuR)
context("CPU vclMatrix algebra")

# set option to use CPU instead of GPU
options(gpuR.default.device.type = "cpu")

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
Aint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
Bint <- matrix(sample(seq(10), ORDER^2, replace=TRUE), nrow=ORDER, ncol=ORDER)
A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
B <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
E <- matrix(rnorm(15), nrow=5)


test_that("CPU vclMatrix Single Precision Matrix multiplication successful", {
    has_cpu_skip()
    
    C <- A %*% B
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    fvclC <- fvclA %*% fvclB
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("CPU vclMatrix Double Precision Matrix multiplication successful", {
    has_cpu_skip()
    
    C <- A %*% B
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    
    dvclC <- dvclA %*% dvclB
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})

test_that("CPU vclMatrix Single Precision Matrix Subtraction successful", {
    has_cpu_skip()
    
    C <- A - B
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    fvclC <- fvclA - fvclB
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("CPU vclMatrix Single Precision Matrix Addition successful", {
    has_cpu_skip()
    
    C <- A + B
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    
    fvclC <- fvclA + fvclB
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
})

test_that("CPU vclMatrix Double Precision Matrix Subtraction successful", {
    has_cpu_skip()
    
    C <- A - B
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    
    dvclC <- dvclA - dvclB
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})

test_that("CPU vclMatrix Double Precision Matrix Addition successful", {
    has_cpu_skip()
    
    C <- A + B
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    
    dvclC <- dvclA + dvclB
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
})

test_that("CPU vclMatrix Single Precision crossprod successful", {
    has_cpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    
    C <- crossprod(X,Y)
    Cs <- crossprod(X)
    
    fvclX <- vclMatrix(X, type="float")
    fvclY <- vclMatrix(Y, type="float")
    fvclZ <- vclMatrix(Z, type="float")
    
    fvclC <- crossprod(fvclX, fvclY)
    fvclCs <- crossprod(fvclX)
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fvclCs[,], Cs, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_error(crossprod(fvclX, fvclZ))
})

test_that("CPU vclMatrix Double Precision crossprod successful", {
    has_cpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(10), nrow=5)
    
    C <- crossprod(X,Y)
    Cs <- crossprod(X)
    
    fvclX <- vclMatrix(X, type="double")
    fvclY <- vclMatrix(Y, type="double")
    fvclZ <- vclMatrix(Z, type="double")
    
    fvclC <- crossprod(fvclX, fvclY)
    fvclCs <- crossprod(fvclX)
    
    expect_is(fvclC, "dvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fvclCs[,], Cs, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_error(crossprod(fvclX, fvclZ))
})

test_that("CPU vclMatrix Single Precision tcrossprod successful", {
    has_cpu_skip()
    
    X <- matrix(rnorm(10), nrow=5)
    Y <- matrix(rnorm(10), nrow=5)
    Z <- matrix(rnorm(15), nrow=5)
    
    C <- tcrossprod(X,Y)
    Cs <- tcrossprod(X)
    
    fvclX <- vclMatrix(X, type="float")
    fvclY <- vclMatrix(Y, type="float")
    fvclZ <- vclMatrix(Z, type="float")
    
    fvclC <- tcrossprod(fvclX, fvclY)
    fvclCs <- tcrossprod(fvclX)
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_equal(fvclCs[,], Cs, tolerance=1e-07, 
                 info="float matrix elements not equivalent") 
    expect_error(tcrossprod(fvclX, fvclZ))
})

test_that("CPU vclMatrix Double Precision tcrossprod successful", {
    has_cpu_skip()
    
    X <- matrix(rnorm(10), nrow=2)
    Y <- matrix(rnorm(10), nrow=2)
    Z <- matrix(rnorm(12), nrow=2)
    
    C <- tcrossprod(X,Y)
    Cs <- tcrossprod(X)
    
    fvclX <- vclMatrix(X, type="double")
    fvclY <- vclMatrix(Y, type="double")
    fvclZ <- vclMatrix(Z, type="double")
    
    fvclC <- tcrossprod(fvclX, fvclY)
    fvclCs <- tcrossprod(fvclX)
    
    expect_is(fvclC, "dvclMatrix")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_equal(fvclCs[,], Cs, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent") 
    expect_error(tcrossprod(fvclX, fvclZ))
})

test_that("CPU vclMatrix Single Precision Matrix Element-Wise Multiplication", {
    has_cpu_skip()
    
    C <- A * B
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    fvclE <- vclMatrix(E, type="float")
    
    fvclC <- fvclA * fvclB
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fvclA * fvclE)
})

test_that("CPU vclMatrix Single Precision Matrix Element-Wise Division", {
    has_cpu_skip()
    
    C <- A / B
    
    fvclA <- vclMatrix(A, type="float")
    fvclB <- vclMatrix(B, type="float")
    fvclE <- vclMatrix(E, type="float")
    
    fvclC <- fvclA / fvclB
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    expect_error(fvclA * fvclE)
})

test_that("CPU vclMatrix Double Precision Matrix Element-Wise Multiplication", {
    has_cpu_skip()
    
    C <- A * B
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    dvclE <- vclMatrix(E, type="double")
    
    dvclC <- dvclA * dvclB
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dvclA * dvclE)
})

test_that("CPU vclMatrix Double Precision Matrix Element-Wise Division", {
    has_cpu_skip()
    
    C <- A / B
    
    dvclA <- vclMatrix(A, type="double")
    dvclB <- vclMatrix(B, type="double")
    dvclE <- vclMatrix(E, type="double")
    
    dvclC <- dvclA / dvclB
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double matrix elements not equivalent")  
    expect_error(dvclA * dvclE)
})

test_that("CPU vclMatrix Single Precision transpose", {
    
    has_cpu_skip()
    
    At <- t(A)
    
    fgpuA <- vclMatrix(A, type="float")
    fgpuAt <- t(fgpuA)
    
    expect_is(fgpuAt, "fvclMatrix")
    expect_equal(fgpuAt[,], At, tolerance=1e-07, 
                 info="transposed float matrix elements not equivalent") 
})

test_that("CPU vclMatrix Double Precision transpose", {
    
    has_cpu_skip()
    
    At <- t(A)
    
    fgpuA <- vclMatrix(A, type="double")
    fgpuAt <- t(fgpuA)
    
    expect_is(fgpuAt, "dvclMatrix")
    expect_equal(fgpuAt[,], At, tolerance=.Machine$double.eps^0.5, 
                 info="transposed double matrix elements not equivalent") 
})


# test_that("CPU vclMatrix Integer Matrix multiplication successful", {
#     
#     has_gpu_skip()
#     
#     Cint <- Aint %*% Bint
#     
#     igpuA <- vclMatrix(Aint, type="integer")
#     igpuB <- vclMatrix(Bint, type="integer")
#     
#     igpuC <- igpuA %*% igpuB
#     
#     expect_equivalent(igpuC[,], Cint, 
#                       info="float matrix elements not equivalent")      
# })
# 
# test_that("CPU vclMatrix Integer Matrix Subtraction successful", {
#     
#     has_gpu_skip()
#     
#     Cint <- Aint - Bint
#     
#     igpuA <- vclMatrix(Aint, type="integer")
#     igpuB <- vclMatrix(Bint, type="integer")
#     
#     igpuC <- igpuA - igpuB
#     
#     expect_is(igpuC, "ivclMatrix")
#     expect_equal(igpuC[,], Cint, 
#                  info="integer matrix elements not equivalent")  
# })
# 
# test_that("CPU vclMatrix Integer Matrix Addition successful", {
#     
#     has_gpu_skip()
#     
#     Cint <- Aint + Bint
#     
#     igpuA <- vclMatrix(Aint, type="integer")
#     igpuB <- vclMatrix(Bint, type="integer")
#     
#     igpuC <- igpuA + igpuB
#     
#     expect_is(igpuC, "ivclMatrix")
#     expect_equal(igpuC[,], Cint,
#                  info="integer matrix elements not equivalent")  
# })

options(gpuR.default.device.type = "gpu")
