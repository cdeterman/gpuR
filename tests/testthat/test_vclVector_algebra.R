library(gpuR)
context("vclVector algebra")

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
Aint <- seq.int(10)
Bint <- sample(seq.int(10), ORDER)
A <- rnorm(ORDER)
B <- rnorm(ORDER)
E <- rnorm(ORDER-1)


test_that("vclVector Single Precision Inner Product successful", {
    
    has_gpu_skip()
    
    C <- A %*% B
    
    fvclA <- vclVector(A, type="float")
    fvclB <- vclVector(B, type="float")
    
    fvclC <- fvclA %*% fvclB
    
    expect_is(fvclC, "matrix")
    expect_equal(fvclC, C, tolerance=1e-06, 
                 info="float vcl vector elements not equivalent")  
})

test_that("vclVector Double Precision Inner Product successful", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A %*% B
    
    dvclA <- vclVector(A, type="double")
    dvclB <- vclVector(B, type="double")
    
    dvclC <- dvclA %*% dvclB
    
    expect_is(dvclC, "matrix")
    expect_equal(dvclC, C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vcl vector elements not equivalent")  
})

test_that("vclVector Single Precision Outer Product successful", {
    
    has_gpu_skip()
    
    C <- A %o% B
    
    fvclA <- vclVector(A, type="float")
    fvclB <- vclVector(B, type="float")
    
    fvclC <- fvclA %o% fvclB
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float vcl vector elements not equivalent")  
})

test_that("vclVector Double Precision Outer Product successful", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A %o% B
    
    dvclA <- vclVector(A, type="double")
    dvclB <- vclVector(B, type="double")
    
    dvclC <- dvclA %o% dvclB
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vcl vector elements not equivalent")  
})

test_that("vclVector Single Precision Vector Subtraction successful", {
    
    has_gpu_skip()
    
    C <- A - B
    
    fvclA <- vclVector(A, type="float")
    fvclB <- vclVector(B, type="float")
    
    fvclC <- fvclA - fvclB
    
    expect_is(fvclC, "fvclVector")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float vcl vector elements not equivalent")  
})

test_that("vclVector Single Precision Vector Addition successful", {
    
    has_gpu_skip()
    
    C <- A + B
    
    fvclA <- vclVector(A, type="float")
    fvclB <- vclVector(B, type="float")
    
    fvclC <- fvclA + fvclB
    
    expect_is(fvclC, "fvclVector")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float vcl vector elements not equivalent")  
})

test_that("vclVector Double Precision Vector Subtraction successful", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A - B
    
    dvclA <- vclVector(A, type="double")
    dvclB <- vclVector(B, type="double")
    
    dvclC <- dvclA - dvclB
    
    expect_is(dvclC, "dvclVector")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vcl vector elements not equivalent")  
})

test_that("vclVector Double Precision Vector Addition successful", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A + B
    
    dvclA <- vclVector(A, type="double")
    dvclB <- vclVector(B, type="double")
    
    dvclC <- dvclA + dvclB
    
    expect_is(dvclC, "dvclVector")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vcl vector elements not equivalent")  
})


test_that("vclVector Single Precision Vector Element-Wise Multiplication", {
    
    has_gpu_skip()
    
    C <- A * B
    
    fvclA <- vclVector(A, type="float")
    fvclB <- vclVector(B, type="float")
    fvclE <- vclVector(E, type="float")
    
    fvclC <- fvclA * fvclB
    
    expect_is(fvclC, "fvclVector")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float vcl vector elements not equivalent")  
    expect_error(fvclA * fvclE)
})

test_that("vclVector Single Precision Vector Element-Wise Division", {
    
    has_gpu_skip()
    
    C <- A / B
    
    fvclA <- vclVector(A, type="float")
    fvclB <- vclVector(B, type="float")
    fvclE <- vclVector(E, type="float")
    
    fvclC <- fvclA / fvclB
    
    expect_is(fvclC, "fvclVector")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float vcl vector elements not equivalent")  
    expect_error(fvclA * fvclE)
})

test_that("vclVector Double Precision Vector Element-Wise Multiplication", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A * B
    
    dvclA <- vclVector(A, type="double")
    dvclB <- vclVector(B, type="double")
    dvclE <- vclVector(E, type="double")
    
    dvclC <- dvclA * dvclB
    
    expect_is(dvclC, "dvclVector")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vcl vector elements not equivalent")  
    expect_error(dvclA * dvclE)
})

test_that("vclVector Double Precision Vector Element-Wise Division", {
    
    has_gpu_skip()
    has_double_skip()
    
    C <- A / B
    
    dvclA <- vclVector(A, type="double")
    dvclB <- vclVector(B, type="double")
    dvclE <- vclVector(E, type="double")
    
    dvclC <- dvclA / dvclB
    
    expect_is(dvclC, "dvclVector")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vcl vector elements not equivalent")  
    expect_error(dvclA * dvclE)
})

