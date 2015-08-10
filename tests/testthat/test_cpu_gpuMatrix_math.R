library(gpuR)
context("gpuMatrix math operations")

# set option to use CPU instead of GPU
options(gpuR.default.device = "cpu")

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
A <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
B <- matrix(rnorm(ORDER^2), nrow=ORDER, ncol=ORDER)
E <- matrix(rnorm(15), nrow=5)


test_that("gpuMatrix Single Precision Matrix Element-Wise Trignometry", {
    
    Sin <- sin(A)
    Asin <- asin(A)
    Hsin <- sinh(A)
    Cos <- cos(A)
    Acos <- acos(A)
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

test_that("gpuMatrix Double Precision Matrix Element-Wise Trignometry", {
    
    Sin <- sin(A)
    Asin <- asin(A)
    Hsin <- sinh(A)
    Cos <- cos(A)
    Acos <- acos(A)
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
                 info="sin float matrix elements not equivalent")  
    expect_equal(fgpuAS[,], Asin, tolerance=.Machine$double.eps ^ 0.5,
                 info="arc sin float matrix elements not equivalent")  
    expect_equal(fgpuHS[,], Hsin, tolerance=.Machine$double.eps ^ 0.5, 
                 info="hyperbolic sin float matrix elements not equivalent")  
    expect_equal(fgpuC[,], Cos, tolerance=.Machine$double.eps ^ 0.5,
                 info="cos float matrix elements not equivalent")    
    expect_equal(fgpuAC[,], Acos, tolerance=.Machine$double.eps ^ 0.5, 
                 info="arc cos float matrix elements not equivalent")    
    expect_equal(fgpuHC[,], Hcos, tolerance=.Machine$double.eps ^ 0.5, 
                 info="hyperbolic cos float matrix elements not equivalent")  
    expect_equal(fgpuT[,], Tan, tolerance=.Machine$double.eps ^ 0.5,
                 info="tan float matrix elements not equivalent")  
    expect_equal(fgpuAT[,], Atan, tolerance=.Machine$double.eps ^ 0.5, 
                 info="arc tan float matrix elements not equivalent")  
    expect_equal(fgpuHT[,], Htan, tolerance=.Machine$double.eps ^ 0.5, 
                 info="hyperbolic tan float matrix elements not equivalent") 
})

options(gpuR.default.device = "gpu")
