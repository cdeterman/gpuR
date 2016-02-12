library(gpuR)
context("CPU vclMatrix math operations")

# set seed
set.seed(123)

# ignore warnings (logs and arc trigs)
options(warn=-1)

# set option to use CPU instead of GPU
options(gpuR.default.device.type = "cpu")

ORDER <- 4

# Base R objects
A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
B <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
E <- matrix(rnorm(15), nrow=5)


test_that("CPU vclMatrix Single Precision Matrix Element-Wise Trignometry", {
    has_cpu_skip()
    
    Sin <- sin(A)
    Asin <- asin(A)
    Hsin <- sinh(A)
    Cos <- cos(A)
    Acos <- acos(A)
    Hcos <- cosh(A)
    Tan <- tan(A) 
    Atan <- atan(A)
    Htan <- tanh(A)
    
    fgpuA <- vclMatrix(A, type="float")
    
    fgpuS <- sin(fgpuA)
    fgpuAS <- asin(fgpuA)
    fgpuHS <- sinh(fgpuA)
    fgpuC <- cos(fgpuA)
    fgpuAC <- acos(fgpuA)
    fgpuHC <- cosh(fgpuA)
    fgpuT <- tan(fgpuA)
    fgpuAT <- atan(fgpuA)
    fgpuHT <- tanh(fgpuA)
    
    expect_is(fgpuC, "fvclMatrix")
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

test_that("CPU vclMatrix Double Precision Matrix Element-Wise Trignometry", {
    has_cpu_skip()
    
    Sin <- sin(A)
    Asin <- asin(A)
    Hsin <- sinh(A)
    Cos <- cos(A)
    Acos <- acos(A)
    Hcos <- cosh(A)
    Tan <- tan(A) 
    Atan <- atan(A)
    Htan <- tanh(A) 
    
    fgpuA <- vclMatrix(A, type="double")
    
    fgpuS <- sin(fgpuA)
    fgpuAS <- asin(fgpuA)
    fgpuHS <- sinh(fgpuA)
    fgpuC <- cos(fgpuA)
    fgpuAC <- acos(fgpuA)
    fgpuHC <- cosh(fgpuA)
    fgpuT <- tan(fgpuA)
    fgpuAT <- atan(fgpuA)
    fgpuHT <- tanh(fgpuA)
    
    expect_is(fgpuC, "dvclMatrix")    
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


test_that("CPU vclMatrix Single Precision Matrix Element-Wise Logs", {
    has_cpu_skip()
    
    R_log <- log(A)
    R_log10 <- log10(A)
    R_log2 <- log(A, base=2)
    
    fgpuA <- vclMatrix(A, type="float")
    
    fgpu_log <- log(fgpuA)
    fgpu_log10 <- log10(fgpuA)
    fgpu_log2 <- log(fgpuA, base=2)
    
    expect_is(fgpu_log, "fvclMatrix")
    expect_is(fgpu_log10, "fvclMatrix")
    expect_is(fgpu_log2, "fvclMatrix")
    expect_equal(fgpu_log[,], R_log, tolerance=1e-07, 
                 info="log float matrix elements not equivalent")  
    expect_equal(fgpu_log10[,], R_log10, tolerance=1e-07, 
                 info="log10 float matrix elements not equivalent")  
    expect_equal(fgpu_log2[,], R_log2, tolerance=1e-07, 
                 info="base log float matrix elements not equivalent") 
})

test_that("CPU vclMatrix Double Precision Matrix Element-Wise Logs", {
    has_cpu_skip()
    
    R_log <- log(A)
    R_log10 <- log10(A)
    R_log2 <- log(A, base=2)
    
    fgpuA <- vclMatrix(A, type="double")
    
    fgpu_log <- log(fgpuA)
    fgpu_log10 <- log10(fgpuA)
    fgpu_log2 <- log(fgpuA, base=2)
    
    expect_is(fgpu_log, "dvclMatrix")
    expect_is(fgpu_log10, "dvclMatrix")
    expect_is(fgpu_log2, "dvclMatrix")
    expect_equal(fgpu_log[,], R_log, tolerance=.Machine$double.eps ^ 0.5, 
                 info="log double matrix elements not equivalent")  
    expect_equal(fgpu_log10[,], R_log10, tolerance=.Machine$double.eps ^ 0.5, 
                 info="log10 double matrix elements not equivalent")  
    expect_equal(fgpu_log2[,], R_log2, tolerance=.Machine$double.eps ^ 0.5, 
                 info="base log double matrix elements not equivalent") 
})


test_that("CPU vclMatrix Single Precision Matrix Exponential", {
    has_cpu_skip()
    
    R_exp <- exp(A)
    
    fgpuA <- vclMatrix(A, type="float")
    
    fgpu_exp <- exp(fgpuA)
    
    expect_is(fgpu_exp, "fvclMatrix")
    expect_equal(fgpu_exp[,], R_exp, tolerance=1e-07, 
                 info="exp float matrix elements not equivalent")  
})

test_that("CPU vclMatrix Double Precision Matrix Exponential", {
    has_cpu_skip()
    
    R_exp <- exp(A)
    
    fgpuA <- vclMatrix(A, type="double")
    
    fgpu_exp <- exp(fgpuA)
    
    expect_is(fgpu_exp, "dvclMatrix")
    expect_equal(fgpu_exp[,], R_exp, tolerance=1e-07, 
                 info="exp double matrix elements not equivalent")  
})


test_that("CPU vclMatrix Single Precision Matrix Absolute Value", {
    
    has_cpu_skip()
    
    R_abs <- abs(A)
    
    fvclA <- vclMatrix(A, type="float")
    
    fvcl_abs <- abs(fvclA)
    
    expect_is(fvcl_abs, "fvclMatrix")
    expect_equal(fvcl_abs[,], R_abs, tolerance=1e-07, 
                 info="abs float matrix elements not equivalent")  
})

test_that("CPU vclMatrix Double Precision Matrix Absolute Value", {
    
    has_cpu_skip()
    
    R_abs <- abs(A)
    
    fvclA <- vclMatrix(A, type="double")
    
    fvcl_abs <- abs(fvclA)
    
    expect_is(fvcl_abs, "dvclMatrix")
    expect_equal(fvcl_abs[,], R_abs, tolerance=.Machine$double.eps^0.5, 
                 info="abs double matrix elements not equivalent")  
})


test_that("CPU vclMatrix Single Precision Maximum/Minimum", {
    
    has_cpu_skip()
    
    R_max <- max(A)
    R_min <- min(A)
    
    fvclA <- vclMatrix(A, type="float")
    
    fvcl_max <- max(fvclA)
    fvcl_min <- min(fvclA)
    
    expect_is(fvcl_max, "numeric")
    expect_equal(fvcl_max, R_max, tolerance=1e-07, 
                 info="max float matrix element not equivalent")  
    expect_equal(fvcl_min, R_min, tolerance=1e-07, 
                 info="min float matrix element not equivalent")  
})

test_that("CPU vclMatrix Double Precision Maximum/Minimum", {
    
    has_cpu_skip()
    
    R_max <- max(A)
    R_min <- min(A)
    
    fvclA <- vclMatrix(A, type="double")
    
    fvcl_max <- max(fvclA)
    fvcl_min <- min(fvclA)
    
    expect_is(fvcl_max, "numeric")
    expect_equal(fvcl_max, R_max, tolerance=.Machine$double.eps^0.5, 
                 info="max double matrix element not equivalent") 
    expect_equal(fvcl_min, R_min, tolerance=.Machine$double.eps^0.5, 
                 info="min double matrix element not equivalent")  
})

# set option back to GPU
options(gpuR.default.device.type = "gpu")
