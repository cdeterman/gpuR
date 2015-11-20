library(gpuR)
context("CPU gpuVector algebra")

# set option to use CPU instead of GPU
options(gpuR.default.device = "cpu")

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
Aint <- seq.int(10)
Bint <- sample(seq.int(10), ORDER)
A <- rnorm(ORDER)
B <- rnorm(ORDER)
E <- rnorm(ORDER-1)


test_that("gpuVector Single Precision Inner Product successful", {
    has_cpu_skip()
    
    C <- A %*% B
    
    fvclA <- gpuVector(A, type="float")
    fvclB <- gpuVector(B, type="float")
    
    fvclC <- fvclA %*% fvclB
    
    expect_is(fvclC, "matrix")
    expect_equal(fvclC, C, tolerance=1e-06, 
                 info="float vcl vector elements not equivalent",
                 check.attributes=FALSE)  
})

test_that("gpuVector Double Precision Inner Product successful", {
    has_cpu_skip()
    
    C <- A %*% B
    
    dvclA <- gpuVector(A, type="double")
    dvclB <- gpuVector(B, type="double")
    
    dvclC <- dvclA %*% dvclB
    
    expect_is(dvclC, "matrix")
    expect_equal(dvclC, C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vcl vector elements not equivalent",
                 check.attributes=FALSE)  
})

test_that("gpuVector Single Precision Outer Product successful", {
    has_cpu_skip()
    
    C <- A %o% B
    
    fvclA <- gpuVector(A, type="float")
    fvclB <- gpuVector(B, type="float")
    
    fvclC <- fvclA %o% fvclB
    
    expect_is(fvclC, "fgpuMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float vcl vector elements not equivalent",
                 check.attributes=FALSE)  
})

test_that("gpuVector Double Precision Outer Product successful", {
    has_cpu_skip()
    
    C <- A %o% B
    
    dvclA <- gpuVector(A, type="double")
    dvclB <- gpuVector(B, type="double")
    
    dvclC <- dvclA %o% dvclB
    
    expect_is(dvclC, "dgpuMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vcl vector elements not equivalent",
                 check.attributes=FALSE)  
})

test_that("gpuVector Single Precision Vector Subtraction successful", {
    has_cpu_skip()
    
    C <- A - B
    
    fvclA <- gpuVector(A, type="float")
    fvclB <- gpuVector(B, type="float")
    
    fvclC <- fvclA - fvclB
    
    expect_is(fvclC, "fgpuVector")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float vcl vector elements not equivalent")  
})

test_that("gpuVector Single Precision Vector Addition successful", {
    has_cpu_skip()
    
    C <- A + B
    
    fvclA <- gpuVector(A, type="float")
    fvclB <- gpuVector(B, type="float")
    
    fvclC <- fvclA + fvclB
    
    expect_is(fvclC, "fgpuVector")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float vcl vector elements not equivalent")  
})

test_that("gpuVector Double Precision Vector Subtraction successful", {
    has_cpu_skip()
    
    C <- A - B
    
    dvclA <- gpuVector(A, type="double")
    dvclB <- gpuVector(B, type="double")
    
    dvclC <- dvclA - dvclB
    
    expect_is(dvclC, "dgpuVector")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vcl vector elements not equivalent")  
})

test_that("gpuVector Double Precision Vector Addition successful", {
    has_cpu_skip()
    
    C <- A + B
    
    dvclA <- gpuVector(A, type="double")
    dvclB <- gpuVector(B, type="double")
    
    dvclC <- dvclA + dvclB
    
    expect_is(dvclC, "dgpuVector")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vcl vector elements not equivalent")  
})


test_that("gpuVector Single Precision Vector Element-Wise Multiplication", {
    has_cpu_skip()
    
    C <- A * B
    
    fvclA <- gpuVector(A, type="float")
    fvclB <- gpuVector(B, type="float")
    fvclE <- gpuVector(E, type="float")
    
    fvclC <- fvclA * fvclB
    
    expect_is(fvclC, "fgpuVector")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float vcl vector elements not equivalent")  
    expect_error(fvclA * fvclE)
})

test_that("gpuVector Single Precision Vector Element-Wise Division", {
    has_cpu_skip()
    
    C <- A / B
    
    fvclA <- gpuVector(A, type="float")
    fvclB <- gpuVector(B, type="float")
    fvclE <- gpuVector(E, type="float")
    
    fvclC <- fvclA / fvclB
    
    expect_is(fvclC, "fgpuVector")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float vcl vector elements not equivalent")  
    expect_error(fvclA * fvclE)
})

test_that("gpuVector Double Precision Vector Element-Wise Multiplication", {
    has_cpu_skip()
    
    C <- A * B
    
    dvclA <- gpuVector(A, type="double")
    dvclB <- gpuVector(B, type="double")
    dvclE <- gpuVector(E, type="double")
    
    dvclC <- dvclA * dvclB
    
    expect_is(dvclC, "dgpuVector")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vcl vector elements not equivalent")  
    expect_error(dvclA * dvclE)
})

test_that("gpuVector Double Precision Vector Element-Wise Division", {
    has_cpu_skip()
    
    C <- A / B
    
    dvclA <- gpuVector(A, type="double")
    dvclB <- gpuVector(B, type="double")
    dvclE <- gpuVector(E, type="double")
    
    dvclC <- dvclA / dvclB
    
    expect_is(dvclC, "dgpuVector")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vcl vector elements not equivalent")  
    expect_error(dvclA * dvclE)
})

options(gpuR.default.device = "gpu")
