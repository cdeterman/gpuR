library(gpuR)
context("CPU gpuMatrix math operations")
options(warn=-1)

current_context <- set_device_context("cpu")

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
B <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
E <- matrix(rnorm(15), nrow=5)


test_that("CPU gpuMatrix Single Precision Matrix Element-Wise Trignometry", {
    has_cpu_skip()
    
    Sin <- sin(A)
    Asin <- suppressWarnings(asin(A))
    Hsin <- sinh(A)
    Cos <- cos(A)
    Acos <- suppressWarnings(acos(A))
    Hcos <- cosh(A)
    Tan <- tan(A) 
    Atan <- atan(A)
    Htan <- tanh(A)
    
    fgpuA <- gpuMatrix(A, type="float")
    
    fgpuS <- sin(fgpuA)
    fgpuAS <- asin(fgpuA)
    fgpuHS <- sinh(fgpuA)
    fgpuC <- cos(fgpuA)
    fgpuAC <- acos(fgpuA)
    fgpuHC <- cosh(fgpuA)
    fgpuT <- tan(fgpuA)
    fgpuAT <- atan(fgpuA)
    fgpuHT <- tanh(fgpuA)
    
    expect_is(fgpuC, "fgpuMatrix")
    expect_equal(fgpuS[,], Sin, tolerance=1e-07, 
                 info="sin float matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuAS[,], Asin, tolerance=1e-07, 
                 info="arc sin float matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuHS[,], Hsin, tolerance=1e-07, 
                 info="hyperbolic sin float matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuC[,], Cos, tolerance=1e-07, 
                 info="cos float matrix elements not equivalent",
                 check.attributes=FALSE)    
    expect_equal(fgpuAC[,], Acos, tolerance=1e-07, 
                 info="arc cos float matrix elements not equivalent",
                 check.attributes=FALSE)    
    expect_equal(fgpuHC[,], Hcos, tolerance=1e-07, 
                 info="hyperbolic cos float matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuT[,], Tan, tolerance=1e-06, 
                 info="tan float matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuAT[,], Atan, tolerance=1e-07, 
                 info="arc tan float matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuHT[,], Htan, tolerance=1e-07, 
                 info="hyperbolic tan float matrix elements not equivalent",
                 check.attributes=FALSE)  
})

test_that("CPU gpuMatrix Double Precision Matrix Element-Wise Trignometry", {
    has_cpu_skip()
    
    Sin <- sin(A)
    Asin <- suppressWarnings(asin(A))
    Hsin <- sinh(A)
    Cos <- cos(A)
    Acos <- suppressWarnings(acos(A))
    Hcos <- cosh(A)
    Tan <- tan(A) 
    Atan <- atan(A)
    Htan <- tanh(A) 
    
    fgpuA <- gpuMatrix(A, type="double")
    
    fgpuS <- sin(fgpuA)
    fgpuAS <- asin(fgpuA)
    fgpuHS <- sinh(fgpuA)
    fgpuC <- cos(fgpuA)
    fgpuAC <- acos(fgpuA)
    fgpuHC <- cosh(fgpuA)
    fgpuT <- tan(fgpuA)
    fgpuAT <- atan(fgpuA)
    fgpuHT <- tanh(fgpuA)
    
    expect_is(fgpuC, "dgpuMatrix")    
    expect_equal(fgpuS[,], Sin, tolerance=.Machine$double.eps ^ 0.5,
                 info="sin matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuAS[,], Asin, tolerance=.Machine$double.eps ^ 0.5,
                 info="arc sin matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuHS[,], Hsin, tolerance=.Machine$double.eps ^ 0.5, 
                 info="hyperbolic sin matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuC[,], Cos, tolerance=.Machine$double.eps ^ 0.5,
                 info="cos matrix elements not equivalent",
                 check.attributes=FALSE)    
    expect_equal(fgpuAC[,], Acos, tolerance=.Machine$double.eps ^ 0.5, 
                 info="arc cos matrix elements not equivalent",
                 check.attributes=FALSE)    
    expect_equal(fgpuHC[,], Hcos, tolerance=.Machine$double.eps ^ 0.5, 
                 info="hyperbolic cos matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuT[,], Tan, tolerance=.Machine$double.eps ^ 0.5,
                 info="tan matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuAT[,], Atan, tolerance=.Machine$double.eps ^ 0.5, 
                 info="arc tan matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuHT[,], Htan, tolerance=.Machine$double.eps ^ 0.5, 
                 info="hyperbolic tan matrix elements not equivalent",
                 check.attributes=FALSE) 
})

test_that("CPU gpuMatrix Single Precision Matrix Element-Wise Logs", {
    has_cpu_skip()
    pocl_check()
    
    R_log <- suppressWarnings(log(A))
    R_log10 <- suppressWarnings(log10(A))
    R_log2 <- suppressWarnings(log(A, base=2))
    
    fgpuA <- gpuMatrix(A, type="float")
    
    fgpu_log <- log(fgpuA)
    fgpu_log10 <- log10(fgpuA)
    fgpu_log2 <- log(fgpuA, base=2)
    
    expect_is(fgpu_log, "fgpuMatrix")
    expect_is(fgpu_log10, "fgpuMatrix")
    expect_is(fgpu_log2, "fgpuMatrix")
    expect_equal(fgpu_log[,], R_log, tolerance=1e-07, 
                 info="log float matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpu_log10[,], R_log10, tolerance=1e-07, 
                 info="log10 float matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpu_log2[,], R_log2, tolerance=1e-07, 
                 info="base log float matrix elements not equivalent",
                 check.attributes=FALSE) 
})

test_that("CPU gpuMatrix Double Precision Matrix Element-Wise Logs", {
    has_cpu_skip()
    pocl_check()
    
    R_log <- suppressWarnings(log(A))
    R_log10 <- suppressWarnings(log10(A))
    R_log2 <- suppressWarnings(log(A, base=2))
    
    fgpuA <- gpuMatrix(A, type="double")
    
    fgpu_log <- log(fgpuA)
    fgpu_log10 <- log10(fgpuA)
    fgpu_log2 <- log(fgpuA, base=2)
    
    expect_is(fgpu_log, "dgpuMatrix")
    expect_is(fgpu_log10, "dgpuMatrix")
    expect_is(fgpu_log2, "dgpuMatrix")
    expect_equal(fgpu_log[,], R_log, tolerance=.Machine$double.eps ^ 0.5, 
                 info="log matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpu_log10[,], R_log10, tolerance=.Machine$double.eps ^ 0.5, 
                 info="log10 matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpu_log2[,], R_log2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="base log matrix elements not equivalent",
                 check.attributes=FALSE) 
})

test_that("CPU gpuMatrix Single Precision Matrix Exponential", {
    has_cpu_skip()
    
    R_exp <- exp(A)
    
    fgpuA <- gpuMatrix(A, type="float")
    
    fgpu_exp <- exp(fgpuA)
    
    expect_is(fgpu_exp, "fgpuMatrix")
    expect_equal(fgpu_exp[,], R_exp, tolerance=1e-07, 
                 info="exp float matrix elements not equivalent",
                 check.attributes=FALSE)  
})

test_that("CPU gpuMatrix Double Precision Matrix Exponential", {
    has_cpu_skip()
    
    R_exp <- exp(A)
    
    fgpuA <- gpuMatrix(A, type="double")
    
    fgpu_exp <- exp(fgpuA)
    
    expect_is(fgpu_exp, "dgpuMatrix")
    expect_equal(fgpu_exp[,], R_exp, tolerance=.Machine$double.eps^0.5, 
                 info="exp double matrix elements not equivalent",
                 check.attributes=FALSE)  
})

test_that("CPU gpuMatrix Single Precision Matrix Absolute Value", {
    
    has_cpu_skip()
    
    R_abs <- abs(A)
    
    fgpuA <- gpuMatrix(A, type="float")
    
    fgpu_abs <- abs(fgpuA)
    
    expect_is(fgpu_abs, "fgpuMatrix")
    expect_equal(fgpu_abs[,], R_abs, tolerance=1e-07, 
                 info="abs float matrix elements not equivalent")  
})

test_that("CPU gpuMatrix Double Precision Matrix Absolute Value", {
    
    has_cpu_skip()
    
    R_abs <- abs(A)
    
    fgpuA <- gpuMatrix(A, type="double")
    
    fgpu_abs <- abs(fgpuA)
    
    expect_is(fgpu_abs, "dgpuMatrix")
    expect_equal(fgpu_abs[,], R_abs, tolerance=.Machine$double.eps^0.5, 
                 info="abs double matrix elements not equivalent")  
})


test_that("CPU gpuMatrix Single Precision Maximum/Minimum", {
    
    has_cpu_skip()
    
    R_max <- max(A)
    R_min <- min(A)
    
    fgpuA <- gpuMatrix(A, type="float")
    
    fgpu_max <- max(fgpuA)
    fgpu_min <- min(fgpuA)
    
    expect_is(fgpu_max, "numeric")
    expect_equal(fgpu_max[], R_max, tolerance=1e-07, 
                 info="max float matrix element not equivalent")  
    expect_equal(fgpu_min[], R_min, tolerance=1e-07, 
                 info="min float matrix element not equivalent")  
})

test_that("CPU gpuMatrix Double Precision Maximum/Minimum", {
    
    has_cpu_skip()
    
    R_max <- max(A)
    R_min <- min(A)
    
    fgpuA <- gpuMatrix(A, type="double")
    
    fgpu_max <- max(fgpuA)
    fgpu_min <- min(fgpuA)
    
    expect_is(fgpu_max, "numeric")
    expect_equal(fgpu_max[], R_max, tolerance=.Machine$double.eps^0.5, 
                 info="max double matrix element not equivalent") 
    expect_equal(fgpu_min[], R_min, tolerance=.Machine$double.eps^0.5, 
                 info="min double matrix element not equivalent")  
})

test_that("CPU gpuMatrix Single Precision pmax/pmin", {
    
    has_cpu_skip()
    
    R_max <- pmax(A, 0)
    R_min <- pmin(A, 0)
    
    fgpuA <- gpuMatrix(A, type="float")
    
    fgpu_max <- pmax(fgpuA, 0)
    fgpu_min <- pmin(fgpuA, 0)
    
    expect_is(fgpu_max, "gpuMatrix")
    expect_equal(fgpu_max[], R_max, tolerance=1e-07, 
                 info="max float matrix element not equivalent")  
    expect_equal(fgpu_min[], R_min, tolerance=1e-07, 
                 info="min float matrix element not equivalent")  
    
    # multiple operations
    R_max <- pmax(A, 0, 1)
    R_min <- pmin(A, 0, 1)
    
    fgpu_max <- pmax(fgpuA, 0, 1)
    fgpu_min <- pmin(fgpuA, 0, 1)
    
    expect_is(fgpu_max, "gpuMatrix")
    expect_equal(fgpu_max[], R_max, tolerance=1e-07, 
                 info="max float matrix element not equivalent")  
    expect_equal(fgpu_min[], R_min, tolerance=1e-07, 
                 info="min float matrix element not equivalent") 
})

test_that("CPU gpuMatrix Double Precision pmax/pmin", {
    
    has_cpu_skip()
    
    R_max <- pmax(A, 0)
    R_min <- pmin(A, 0)
    
    fgpuA <- gpuMatrix(A, type="double")
    
    fgpu_max <- pmax(fgpuA, 0)
    fgpu_min <- pmin(fgpuA, 0)
    
    expect_is(fgpu_max, "gpuMatrix")
    expect_equal(fgpu_max[], R_max, tolerance=.Machine$double.eps^0.5, 
                 info="max double matrix element not equivalent") 
    expect_equal(fgpu_min[], R_min, tolerance=.Machine$double.eps^0.5, 
                 info="min double matrix element not equivalent")  
    
    # multiple operations
    R_max <- pmax(A, 0, 1)
    R_min <- pmin(A, 0, 1)
    
    fgpu_max <- pmax(fgpuA, 0, 1)
    fgpu_min <- pmin(fgpuA, 0, 1)
    
    expect_is(fgpu_max, "gpuMatrix")
    expect_equal(fgpu_max[], R_max, tolerance=.Machine$double.eps^0.5, 
                 info="max double matrix element not equivalent") 
    expect_equal(fgpu_min[], R_min, tolerance=.Machine$double.eps^0.5, 
                 info="min double matrix element not equivalent")  
})

test_that("CPU gpuMatrix Single Precision Matrix sqrt", {
    has_cpu_skip()
    
    R_sqrt <- sqrt(abs(A))
    
    fgpuA <- gpuMatrix(abs(A), type="float")
    
    fgpu_sqrt <- sqrt(fgpuA)
    
    expect_is(fgpu_sqrt, "fgpuMatrix")
    expect_equal(fgpu_sqrt[,], R_sqrt, tolerance=1e-07, 
                 info="sqrt float matrix elements not equivalent",
                 check.attributes=FALSE)  
})

test_that("CPU gpuMatrix Double Precision Matrix sqrt", {
    has_cpu_skip()
    
    R_sqrt <- sqrt(abs(A))
    
    fgpuA <- gpuMatrix(abs(A), type="double")
    
    fgpu_sqrt <- sqrt(fgpuA)
    
    expect_is(fgpu_sqrt, "dgpuMatrix")
    expect_equal(fgpu_sqrt[,], R_sqrt, tolerance=.Machine$double.eps^0.5, 
                 info="sqrt double matrix elements not equivalent",
                 check.attributes=FALSE)  
})

# test_that("CPU gpuMatrix Integer Precision Matrix sign", {
#     has_cpu_skip()
#     
#     Ai <- matrix(seq.int(16), 4, 4) * sample(c(-1, 1), 16, replace = TRUE)
#     R_sign <- sign(Ai)
#     
#     fgpuA <- gpuMatrix(Ai, type="integer")
#     
#     fgpu_sign <- sign(fgpuA)
#     
#     expect_is(fgpu_sign, "igpuMatrix")
#     expect_equivalent(fgpu_sign[,], R_sign, 
#                  info="sign integer matrix elements not equivalent",
#                  check.attributes=FALSE)  
# })

test_that("CPU gpuMatrix Single Precision Matrix sign", {
    has_cpu_skip()
    
    R_sign <- sign(A)
    
    fgpuA <- gpuMatrix(A, type="float")
    
    fgpu_sign <- sign(fgpuA)
    
    expect_is(fgpu_sign, "fgpuMatrix")
    expect_equal(fgpu_sign[,], R_sign, tolerance=1e-07, 
                 info="sign float matrix elements not equivalent",
                 check.attributes=FALSE)  
})

test_that("CPU gpuMatrix Double Precision Matrix sign", {
    has_cpu_skip()
    
    R_sign <- sign(A)
    
    fgpuA <- gpuMatrix(A, type="double")
    
    fgpu_sign <- sign(fgpuA)
    
    expect_is(fgpu_sign, "dgpuMatrix")
    expect_equal(fgpu_sign[,], R_sign, tolerance=.Machine$double.base^0.5, 
                 info="sign double matrix elements not equivalent",
                 check.attributes=FALSE)  
})

options(warn=0)

setContext(current_context)