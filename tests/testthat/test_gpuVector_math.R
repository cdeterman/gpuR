library(gpuR)
context("gpuVector math operations")

if(detectGPUs() >= 1){
    current_context <- set_device_context("gpu")    
}else{
    current_context <- currentContext()
}

# set seed
set.seed(123)

# ignore warnings (logs and arc trigs)
options(warn=-1)

ORDER <- 4

# Base R objects
A <- rnorm(ORDER)
B <- rnorm(ORDER)


test_that("gpuVector Single Precision Element-Wise Trignometry", {
    
    has_gpu_skip()
    
    Sin <- sin(A)
    Asin <- suppressWarnings(asin(A))
    Hsin <- sinh(A)
    Cos <- cos(A)
    Acos <- suppressWarnings(acos(A))
    Hcos <- cosh(A)
    Tan <- tan(A) 
    Atan <- atan(A)
    Htan <- tanh(A)
    
    fgpuA <- gpuVector(A, type="float")
    
    fgpuS <- sin(fgpuA)
    fgpuAS <- asin(fgpuA)
    fgpuHS <- sinh(fgpuA)
    fgpuC <- cos(fgpuA)
    fgpuAC <- acos(fgpuA)
    fgpuHC <- cosh(fgpuA)
    fgpuT <- tan(fgpuA)
    fgpuAT <- atan(fgpuA)
    fgpuHT <- tanh(fgpuA)
    
    expect_is(fgpuS, "fgpuVector")
    expect_equal(fgpuS[,], Sin, tolerance=1e-07, 
                 info="sin float matrix elements not equivalent")  
    expect_equal(fgpuAS[,], Asin, tolerance=1e-07, 
                 info="arc sin float matrix elements not equivalent")  
    expect_equal(fgpuHS[,], Hsin, tolerance=1e-07, 
                 info="hyperbolic sin float matrix elements not equivalent")  
    expect_equal(fgpuC[,], Cos, tolerance=1e-07, 
                 info="cos float matrix elements not equivalent")    
    expect_equal(fgpuAC[,], Acos, tolerance=1e-07, 
                 info="arc cos float matrix elements not equivalent")    
    expect_equal(fgpuHC[,], Hcos, tolerance=1e-07, 
                 info="hyperbolic cos float matrix elements not equivalent")  
    expect_equal(fgpuT[,], Tan, tolerance=1e-06, 
                 info="tan float matrix elements not equivalent")  
    expect_equal(fgpuAT[,], Atan, tolerance=1e-07, 
                 info="arc tan float matrix elements not equivalent")  
    expect_equal(fgpuHT[,], Htan, tolerance=1e-07, 
                 info="hyperbolic tan float matrix elements not equivalent")  
})

test_that("gpuVector Double Precision Element-Wise Trignometry", {
    
    has_gpu_skip()
    has_double_skip()
    
    Sin <- sin(A)
    Asin <- suppressWarnings(asin(A))
    Hsin <- sinh(A)
    Cos <- cos(A)
    Acos <- suppressWarnings(acos(A))
    Hcos <- cosh(A)
    Tan <- tan(A) 
    Atan <- atan(A)
    Htan <- tanh(A) 
    
    fgpuA <- gpuVector(A, type="double")
    
    fgpuS <- sin(fgpuA)
    fgpuAS <- asin(fgpuA)
    fgpuHS <- sinh(fgpuA)
    fgpuC <- cos(fgpuA)
    fgpuAC <- acos(fgpuA)
    fgpuHC <- cosh(fgpuA)
    fgpuT <- tan(fgpuA)
    fgpuAT <- atan(fgpuA)
    fgpuHT <- tanh(fgpuA)
    
    expect_is(fgpuS, "dgpuVector")    
    expect_equal(fgpuS[,], Sin, tolerance=.Machine$double.eps ^ 0.5,
                 info="sin double matrix elements not equivalent")  
    expect_equal(fgpuAS[,], Asin, tolerance=.Machine$double.eps ^ 0.5,
                 info="arc sin double matrix elements not equivalent")  
    expect_equal(fgpuHS[,], Hsin, tolerance=.Machine$double.eps ^ 0.5, 
                 info="hyperbolic sin double matrix elements not equivalent")  
    expect_equal(fgpuC[,], Cos, tolerance=.Machine$double.eps ^ 0.5,
                 info="cos double matrix elements not equivalent")    
    expect_equal(fgpuAC[,], Acos, tolerance=.Machine$double.eps ^ 0.5, 
                 info="arc cos double matrix elements not equivalent")    
    expect_equal(fgpuHC[,], Hcos, tolerance=.Machine$double.eps ^ 0.5, 
                 info="hyperbolic cos double matrix elements not equivalent")  
    expect_equal(fgpuT[,], Tan, tolerance=.Machine$double.eps ^ 0.5,
                 info="tan double matrix elements not equivalent")  
    expect_equal(fgpuAT[,], Atan, tolerance=.Machine$double.eps ^ 0.5, 
                 info="arc tan double matrix elements not equivalent")  
    expect_equal(fgpuHT[,], Htan, tolerance=.Machine$double.eps ^ 0.5, 
                 info="hyperbolic tan double matrix elements not equivalent") 
})

test_that("gpuVector Single Precision Element-Wise Logs", {
    
    has_gpu_skip()
    pocl_check()
    
    R_log <- suppressWarnings(log(A))
    R_log10 <- suppressWarnings(log10(A))
    R_log2 <- suppressWarnings(log(A, base=2))
    
    fgpuA <- gpuVector(A, type="float")
    
    fgpu_log <- log(fgpuA)
    fgpu_log10 <- log10(fgpuA)
    fgpu_log2 <- log(fgpuA, base=2)
    
    expect_is(fgpu_log, "fgpuVector")
    expect_is(fgpu_log10, "fgpuVector")
    expect_is(fgpu_log2, "fgpuVector")
    expect_equal(fgpu_log[,], R_log, tolerance=1e-07, 
                 info="log float matrix elements not equivalent")  
    expect_equal(fgpu_log10[,], R_log10, tolerance=1e-07, 
                 info="log10 float matrix elements not equivalent")  
    expect_equal(fgpu_log2[,], R_log2, tolerance=1e-06, 
                 info="base log float matrix elements not equivalent") 
})

test_that("gpuVector Double Precision Element-Wise Logs", {
    
    has_gpu_skip()
    has_double_skip()
    pocl_check()
    
    R_log <- suppressWarnings(log(A))
    R_log10 <- suppressWarnings(log10(A))
    R_log2 <- suppressWarnings(log(A, base=2))
    
    fgpuA <- gpuVector(A, type="double")
    
    fgpu_log <- log(fgpuA)
    fgpu_log10 <- log10(fgpuA)
    fgpu_log2 <- log(fgpuA, base=2)
    
    expect_is(fgpu_log, "dgpuVector")
    expect_is(fgpu_log10, "dgpuVector")
    expect_is(fgpu_log2, "dgpuVector")
    expect_equal(fgpu_log[,], R_log, tolerance=.Machine$double.eps ^ 0.5, 
                 info="log double matrix elements not equivalent")  
    expect_equal(fgpu_log10[,], R_log10, tolerance=.Machine$double.eps ^ 0.5, 
                 info="log10 double matrix elements not equivalent")  
    expect_equal(fgpu_log2[,], R_log2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="base log double matrix elements not equivalent") 
})

test_that("gpuVector Single Precision Exponential", {
    
    has_gpu_skip()
    
    R_exp <- exp(A)
    
    fgpuA <- gpuVector(A, type="float")
    
    fgpu_exp <- exp(fgpuA)
    
    expect_is(fgpu_exp, "fgpuVector")
    expect_equal(fgpu_exp[,], R_exp, tolerance=1e-07, 
                 info="exp float vector elements not equivalent")  
})

test_that("gpuVector Double Precision Exponential", {
    
    has_gpu_skip()
    has_double_skip()
    
    R_exp <- exp(A)
    
    fgpuA <- gpuVector(A, type="double")
    
    fgpu_exp <- exp(fgpuA)
    
    expect_is(fgpu_exp, "dgpuVector")
    expect_equal(fgpu_exp[,], R_exp, tolerance=.Machine$double.eps^0.5, 
                 info="exp double vector elements not equivalent")  
})

test_that("gpuVector Single Precision Absolute Value", {
    
    has_gpu_skip()
    
    R_abs <- abs(A)
    
    fgpuA <- gpuVector(A, type="float")
    
    fgpu_abs <- abs(fgpuA)
    
    expect_is(fgpu_abs, "fgpuVector")
    expect_equal(fgpu_abs[,], R_abs, tolerance=1e-07, 
                 info="abs float vector elements not equivalent")  
})

test_that("gpuVector Double Precision Absolute Value", {
    
    has_gpu_skip()
    has_double_skip()
    
    R_abs <- abs(A)
    
    fgpuA <- gpuVector(A, type="double")
    
    fgpu_abs <- abs(fgpuA)
    
    expect_is(fgpu_abs, "dgpuVector")
    expect_equal(fgpu_abs[,], R_abs, tolerance=.Machine$double.eps^0.5, 
                 info="abs double vector elements not equivalent")  
})

test_that("gpuVector Single Precision Maximum/Minimum", {
    
    has_gpu_skip()
    
    R_max <- max(A)
    R_min <- min(A)
    
    fgpuA <- gpuVector(A, type="float")
    
    fgpu_max <- max(fgpuA)
    fgpu_min <- min(fgpuA)
    
    expect_is(fgpu_max, "numeric")
    expect_equal(fgpu_max, R_max, tolerance=1e-07, 
                 info="max float vector element not equivalent")  
    expect_equal(fgpu_min, R_min, tolerance=1e-07, 
                 info="min float vector element not equivalent")  
})

test_that("gpuVector Double Precision Maximum/Minimum", {
    
    has_gpu_skip()
    has_double_skip()
    
    R_max <- max(A)
    R_min <- min(A)
    
    fgpuA <- gpuVector(A, type="double")
    
    fgpu_max <- max(fgpuA)
    fgpu_min <- min(fgpuA)
    
    expect_is(fgpu_max, "numeric")
    expect_equal(fgpu_max, R_max, tolerance=.Machine$double.eps^0.5, 
                 info="max double vector element not equivalent") 
    expect_equal(fgpu_min, R_min, tolerance=.Machine$double.eps^0.5, 
                 info="min double vector element not equivalent")  
})

test_that("gpuVector Single Precision pmax/pmin", {
    
    has_gpu_skip()
    
    R_max <- pmax(A, 0)
    R_min <- pmin(A, 0)
    
    fgpuA <- gpuVector(A, type="float")
    
    fgpu_max <- pmax(fgpuA, 0)
    fgpu_min <- pmin(fgpuA, 0)
    
    expect_is(fgpu_max, "fgpuVector")
    expect_equal(fgpu_max[], R_max, tolerance=1e-07, 
                 info="max float vector element not equivalent")  
    expect_equal(fgpu_min[], R_min, tolerance=1e-07, 
                 info="min float vector element not equivalent")  
    
    # multiple operations
    R_max <- pmax(A, 0, .5)
    R_min <- pmin(A, 0, -.2)
    
    fgpu_max <- pmax(fgpuA, 0, 0.5)
    fgpu_min <- pmin(fgpuA, 0, -.2)
    
    expect_is(fgpu_max, "fgpuVector")
    expect_equal(fgpu_max[], R_max, tolerance=1e-07, 
                 info="max float vector element not equivalent")  
    expect_equal(fgpu_min[], R_min, tolerance=1e-07, 
                 info="min float vector element not equivalent") 
})

test_that("gpuVector Double Precision pmax/pmin", {
    
    has_gpu_skip()
    has_double_skip()
    
    R_max <- pmax(A, 0)
    R_min <- pmin(A, 0)
    
    fgpuA <- gpuVector(A, type="double")
    
    fgpu_max <- pmax(fgpuA, 0)
    fgpu_min <- pmin(fgpuA, 0)
    
    expect_is(fgpu_max, "dgpuVector")
    expect_equal(fgpu_max[], R_max, tolerance=.Machine$double.eps^0.5, 
                 info="max double vector element not equivalent") 
    expect_equal(fgpu_min[], R_min, tolerance=.Machine$double.eps^0.5, 
                 info="min double vector element not equivalent")  
    
    
    # multiple operations
    R_max <- pmax(A, 0, .5)
    R_min <- pmin(A, 0, -.2)
    
    fgpu_max <- pmax(fgpuA, 0, 0.5)
    fgpu_min <- pmin(fgpuA, 0, -.2)
    
    expect_is(fgpu_max, "dgpuVector")
    expect_equal(fgpu_max[], R_max, tolerance=.Machine$double.eps^0.5, 
                 info="max double vector element not equivalent") 
    expect_equal(fgpu_min[], R_min, tolerance=.Machine$double.eps^0.5, 
                 info="min double vector element not equivalent")  
})

test_that("gpuVector Single Precision sqrt", {
    
    has_gpu_skip()
    
    R_sqrt <- sqrt(abs(A))
    
    fgpuA <- gpuVector(abs(A), type="float")
    
    fgpu_sqrt <- sqrt(fgpuA)
    
    expect_is(fgpu_sqrt, "fgpuVector")
    expect_equal(fgpu_sqrt[,], R_sqrt, tolerance=1e-07, 
                 info="sqrt float vector elements not equivalent")  
})

test_that("gpuVector Double Precision sqrt", {
    
    has_gpu_skip()
    has_double_skip()
    
    R_sqrt <- sqrt(abs(A))
    
    fgpuA <- gpuVector(abs(A), type="double")
    
    fgpu_sqrt <- sqrt(fgpuA)
    
    expect_is(fgpu_sqrt, "dgpuVector")
    expect_equal(fgpu_sqrt[,], R_sqrt, tolerance=.Machine$double.eps^0.5, 
                 info="sqrt double vector elements not equivalent")  
})

test_that("gpuVector Integer Precision Matrix sign", {
    has_gpu_skip()
    
    Ai <- seq.int(16) * sample(c(-1, 1), 16, replace = TRUE)
    
    R_sign <- sign(Ai)
    
    fgpuA <- gpuVector(Ai, type="integer")
    
    fgpu_sign <- sign(fgpuA)
    
    expect_is(fgpu_sign, "igpuVector")
    expect_equal(fgpu_sign[,], R_sign, 
                 info="sign integer matrix elements not equivalent",
                 check.attributes=FALSE)  
})

test_that("gpuVector Single Precision Matrix sign", {
    has_gpu_skip()
    
    R_sign <- sign(A)
    
    fgpuA <- gpuVector(A, type="float")
    
    fgpu_sign <- sign(fgpuA)
    
    expect_is(fgpu_sign, "fgpuVector")
    expect_equal(fgpu_sign[,], R_sign, tolerance=1e-07, 
                 info="sign float matrix elements not equivalent",
                 check.attributes=FALSE)  
})

test_that("gpuVector Double Precision Matrix sign", {
    has_gpu_skip()
    has_double_skip()
    
    R_sign <- sign(A)
    
    fgpuA <- gpuVector(A, type="double")
    
    fgpu_sign <- sign(fgpuA)
    
    expect_is(fgpu_sign, "dgpuVector")
    expect_equal(fgpu_sign[,], R_sign, tolerance=.Machine$double.base^0.5, 
                 info="sign double matrix elements not equivalent",
                 check.attributes=FALSE)  
})

setContext(current_context)
