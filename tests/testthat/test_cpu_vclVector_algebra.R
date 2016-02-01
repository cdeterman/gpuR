library(gpuR)
context("CPU vclVector algebra")

# set option to use CPU instead of GPU
options(gpuR.default.device.type = "cpu")

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
Aint <- seq.int(10)
Bint <- sample(seq.int(10), ORDER)
A <- rnorm(ORDER)
B <- rnorm(ORDER)
E <- rnorm(ORDER-1)

# Single Precision Tests

test_that("CPU vclVector Single Precision Vector Addition ", {
    
    has_cpu_skip()
    
    C <- A + B
    
    fvclA <- vclVector(A, type="float")
    fvclB <- vclVector(B, type="float")
    
    fvclC <- fvclA + fvclB
    
    expect_is(fvclC, "fvclVector")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float vcl vector elements not equivalent")  
})

test_that("CPU vclVector Single Precision Scalar Addition", {
    
    has_cpu_skip()
    
    C <- A + 1
    C2 <- 1 + A
    
    fvclA <- vclVector(A, type="float")
    
    fvclC <- fvclA + 1
    fvclC2 <- 1 + fvclA
    
    expect_is(fvclC, "fvclVector")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float vector elements not equivalent") 
    expect_is(fvclC2, "fvclVector")
    expect_equal(fvclC2[,], C2, tolerance=1e-07, 
                 info="float vector elements not equivalent") 
})

test_that("CPU vclVector Single Precision Vector Subtraction ", {
    
    has_cpu_skip()
    
    C <- A - B
    
    fvclA <- vclVector(A, type="float")
    fvclB <- vclVector(B, type="float")
    
    fvclC <- fvclA - fvclB
    
    expect_is(fvclC, "fvclVector")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float vcl vector elements not equivalent")  
})

test_that("gpuVector Single Precision Scalar Matrix Subtraction", {
    
    has_cpu_skip()
    
    C <- A - 1
    C2 <- 1 - A
    
    fvclA <- vclVector(A, type="float")
    
    fvclC <- fvclA - 1    
    fvclC2 <- 1 - fvclA
    
    expect_is(fvclC, "fvclVector")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float vector elements not equivalent") 
    expect_is(fvclC2, "fvclVector")
    expect_equal(fvclC2[,], C2, tolerance=1e-07, 
                 info="float vector elements not equivalent") 
})

test_that("CPU vclVector Single Precision Unary Vector Subtraction", {
    
    has_cpu_skip()
    
    C <- -A
    
    fvclA <- vclVector(A, type="float")
    
    fvclC <- -fvclA
    
    expect_is(fvclC, "fvclVector")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float vector elements not equivalent") 
})

test_that("CPU vclVector Single Precision Vector Element-Wise Multiplication", {
    
    has_cpu_skip()
    
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

test_that("CPU vclVector Single Precision Scalar Vector Multiplication", {
    
    has_cpu_skip()
    
    C <- A * 2
    C2 <- 2 * A
    
    dvclA <- vclVector(A, type="float")
    
    dvclC <- dvclA * 2
    dvclC2 <- 2 * dvclA
    
    expect_is(dvclC, "fvclVector")
    expect_equal(dvclC[,], C, tolerance=1e-07, 
                 info="float vector elements not equivalent") 
    expect_is(dvclC2, "fvclVector")
    expect_equal(dvclC2[,], C2, tolerance=1e-07, 
                 info="float vector elements not equivalent") 
})

test_that("CPU vclVector Single Precision Vector Element-Wise Division", {
    
    has_cpu_skip()
    
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

test_that("CPU vclVector Single Precision Scalar Division", {
    
    has_cpu_skip()
    
    C <- A/2
    C2 <- 2/A
    
    dvclA <- vclVector(A, type="float")
    
    dvclC <- dvclA/2
    dvclC2 <- 2/dvclA
    
    expect_is(dvclC, "fvclVector")
    expect_equal(dvclC[,], C, tolerance=1e-07, 
                 info="float vector elements not equivalent") 
    expect_is(dvclC2, "fvclVector")
    expect_equal(dvclC2[,], C2, tolerance=1e-07, 
                 info="float vector elements not equivalent") 
})

test_that("CPU vclVector Single Precision Vector Element-Wise Power", {
    
    has_cpu_skip()
    
    C <- A ^ B
    
    fvclA <- vclVector(A, type="float")
    fvclB <- vclVector(B, type="float")
    fvclE <- vclVector(E, type="float")
    
    fvclC <- fvclA ^ fvclB
    
    expect_is(fvclC, "fvclVector")
    expect_equal(fvclC[,], C, tolerance=1e-06, 
                 info="float vcl vector elements not equivalent")  
    expect_error(fvclA * fvclE)
})

test_that("CPU vclVector Single Precision Scalar Power", {
    
    has_cpu_skip()
    
    C <- A^2
    C2 <- 2^A
    
    dvclA <- vclVector(A, type="float")
    
    dvclC <- dvclA^2
    dvclC2 <- 2^dvclA
    
    expect_is(dvclC, "fvclVector")
    expect_equal(dvclC[,], C, tolerance=1e-07, 
                 info="float vector elements not equivalent") 
    expect_is(dvclC2, "fvclVector")
    expect_equal(dvclC2[,], C2, tolerance=1e-07, 
                 info="float vector elements not equivalent") 
})

test_that("CPU vclVector Single Precision Inner Product ", {
    
    has_cpu_skip()
    
    C <- A %*% B
    
    fvclA <- vclVector(A, type="float")
    fvclB <- vclVector(B, type="float")
    
    fvclC <- fvclA %*% fvclB
    
    expect_is(fvclC, "matrix")
    expect_equal(fvclC, C, tolerance=1e-06, 
                 info="float vcl vector elements not equivalent")  
})

test_that("CPU vclVector Single Precision Outer Product ", {
    
    has_cpu_skip()
    
    C <- A %o% B
    
    fvclA <- vclVector(A, type="float")
    fvclB <- vclVector(B, type="float")
    
    fvclC <- fvclA %o% fvclB
    
    expect_is(fvclC, "fvclMatrix")
    expect_equal(fvclC[,], C, tolerance=1e-07, 
                 info="float vcl vector elements not equivalent")  
})

# Double Precision Tests

test_that("CPU vclVector Double Precision Vector Addition ", {
    
    has_cpu_skip()
    
    C <- A + B
    
    dvclA <- vclVector(A, type="double")
    dvclB <- vclVector(B, type="double")
    
    dvclC <- dvclA + dvclB
    
    expect_is(dvclC, "dvclVector")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vcl vector elements not equivalent")  
})

test_that("CPU vclVector Double Precision Scalar Addition", {
    
    has_cpu_skip()
    
    C <- A + 1
    C2 <- 1 + A
    
    fvclA <- vclVector(A, type="double")
    
    fvclC <- fvclA + 1
    fvclC2 <- 1 + fvclA
    
    expect_is(fvclC, "dvclVector")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vector elements not equivalent") 
    expect_is(fvclC2, "dvclVector")
    expect_equal(fvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vector elements not equivalent") 
})

test_that("CPU vclVector Double Precision Vector Subtraction ", {
    
    has_cpu_skip()
    
    C <- A - B
    
    dvclA <- vclVector(A, type="double")
    dvclB <- vclVector(B, type="double")
    
    dvclC <- dvclA - dvclB
    
    expect_is(dvclC, "dvclVector")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vcl vector elements not equivalent")  
})

test_that("CPU vclVector Double Precision Scalar Matrix Subtraction", {
    
    has_cpu_skip()
    
    C <- A - 1
    C2 <- 1 - A
    
    fvclA <- vclVector(A, type="double")
    
    fvclC <- fvclA - 1    
    fvclC2 <- 1 - fvclA
    
    expect_is(fvclC, "dvclVector")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vector elements not equivalent") 
    expect_is(fvclC2, "dvclVector")
    expect_equal(fvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vector elements not equivalent") 
})

test_that("CPU vclVector Double Precision Unary Vector Subtraction", {
    
    has_cpu_skip()
    
    C <- -A
    
    fvclA <- vclVector(A, type="double")
    
    fvclC <- -fvclA
    
    expect_is(fvclC, "dvclVector")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vector elements not equivalent") 
})

test_that("CPU vclVector Double Precision Vector Element-Wise Multiplication", {
    
    has_cpu_skip()
    
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

test_that("CPU vclVector Double Precision Scalar Multiplication", {
    
    has_cpu_skip()
    
    C <- A * 2
    C2 <- 2 * A
    
    dvclA <- vclVector(A, type="double")
    
    dvclC <- dvclA * 2
    dvclC2 <- 2 * dvclA
    
    expect_is(dvclC, "dvclVector")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vector elements not equivalent") 
    expect_is(dvclC2, "dvclVector")
    expect_equal(dvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vector elements not equivalent") 
})

test_that("CPU vclVector Double Precision Vector Element-Wise Division", {
    
    has_cpu_skip()
    
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

test_that("CPU vclVector Double Precision Scalar Division", {
    
    has_cpu_skip()
    
    C <- A/2
    C2 <- 2/A
    
    dvclA <- vclVector(A, type="double")
    
    dvclC <- dvclA/2
    dvclC2 <- 2/dvclA
    
    expect_is(dvclC, "dvclVector")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vector elements not equivalent") 
    expect_is(dvclC2, "dvclVector")
    expect_equal(dvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vector elements not equivalent") 
})

test_that("CPU vclVector Double Precision Vector Element-Wise Power", {
    
    has_cpu_skip()
    
    C <- A ^ B
    
    fvclA <- vclVector(A, type="double")
    fvclB <- vclVector(B, type="double")
    fvclE <- vclVector(E, type="double")
    
    fvclC <- fvclA ^ fvclB
    
    expect_is(fvclC, "dvclVector")
    expect_equal(fvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vector elements not equivalent")  
    expect_error(fvclA * fvclE)
})

test_that("CPU vclVector Double Precision Scalar Power", {
    
    has_cpu_skip()
    
    C <- A^2
    C2 <- 2^A
    
    dvclA <- vclVector(A, type="double")
    
    dvclC <- dvclA^2
    dvclC2 <- 2^dvclA
    
    expect_is(dvclC, "dvclVector")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vector elements not equivalent") 
    expect_is(dvclC2, "dvclVector")
    expect_equal(dvclC2[,], C2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vector elements not equivalent") 
})

test_that("CPU vclVector Double Precision Inner Product ", {
    
    has_cpu_skip()
    
    C <- A %*% B
    
    dvclA <- vclVector(A, type="double")
    dvclB <- vclVector(B, type="double")
    
    dvclC <- dvclA %*% dvclB
    
    expect_is(dvclC, "matrix")
    expect_equal(dvclC, C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vcl vector elements not equivalent")  
})

test_that("CPU vclVector Double Precision Outer Product ", {
    
    has_cpu_skip()
    
    C <- A %o% B
    
    dvclA <- vclVector(A, type="double")
    dvclB <- vclVector(B, type="double")
    
    dvclC <- dvclA %o% dvclB
    
    expect_is(dvclC, "dvclMatrix")
    expect_equal(dvclC[,], C, tolerance=.Machine$double.eps ^ 0.5, 
                 info="double vcl vector elements not equivalent")  
})

options(gpuR.default.device.type = "gpu")
