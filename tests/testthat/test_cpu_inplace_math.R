library(gpuR)
context("CPU Inplace Math Operations")

current_context <- set_device_context("cpu")

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
A_vec <- rnorm(ORDER^2)
B_vec <- rnorm(ORDER^2)
Aint_vec <- sample(seq(10), ORDER^2, replace=TRUE)
Bint_vec <- sample(seq(10), ORDER^2, replace=TRUE)
Aint <- matrix(Aint_vec, nrow=ORDER, ncol=ORDER)
Bint <- matrix(Bint_vec, nrow=ORDER, ncol=ORDER)
A <- matrix(A_vec, nrow=ORDER, ncol=ORDER)
B <- matrix(B_vec, nrow=ORDER, ncol=ORDER)


test_that("CPU Inplace gpuMatrix Single Precision Matrix Element-Wise Trignometry", {
    
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
    
    fgpuS <- gpuMatrix(A, type="float")
    inplace(`sin`, fgpuS)
    fgpuAS <- gpuMatrix(A, type="float")
    inplace(`asin`, fgpuAS)
    fgpuHS <- gpuMatrix(A, type="float")
    inplace(`sinh`, fgpuHS)
    fgpuC <- gpuMatrix(A, type="float")
    inplace(`cos`, fgpuC)
    fgpuAC <- gpuMatrix(A, type="float")
    inplace(`acos`, fgpuAC)
    fgpuHC <- gpuMatrix(A, type="float")
    inplace(`cosh`, fgpuHC)
    fgpuT <- gpuMatrix(A, type="float")
    inplace(`tan`, fgpuT)
    fgpuAT <- gpuMatrix(A, type="float")
    inplace(`atan`,fgpuAT)
    fgpuHT <- gpuMatrix(A, type="float")
    inplace(`tanh`, fgpuHT)
    
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

test_that("CPU Inplace gpuMatrix Double Precision Matrix Element-Wise Trignometry", {
    
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
    
    fgpuS <- gpuMatrix(A, type="double")
    inplace(`sin`, fgpuS)
    fgpuAS <- gpuMatrix(A, type="double")
    inplace(`asin`, fgpuAS)
    fgpuHS <- gpuMatrix(A, type="double")
    inplace(`sinh`, fgpuHS)
    fgpuC <- gpuMatrix(A, type="double")
    inplace(`cos`, fgpuC)
    fgpuAC <- gpuMatrix(A, type="double")
    inplace(`acos`, fgpuAC)
    fgpuHC <- gpuMatrix(A, type="double")
    inplace(`cosh`, fgpuHC)
    fgpuT <- gpuMatrix(A, type="double")
    inplace(`tan`, fgpuT)
    fgpuAT <- gpuMatrix(A, type="double")
    inplace(`atan`,fgpuAT)
    fgpuHT <- gpuMatrix(A, type="double")
    inplace(`tanh`, fgpuHT)
    
    expect_equal(fgpuS[,], Sin, tolerance=.Machine$double.eps^0.5, 
                 info="sin double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuAS[,], Asin, tolerance=.Machine$double.eps^0.5, 
                 info="arc sin double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuHS[,], Hsin, tolerance=.Machine$double.eps^0.5, 
                 info="hyperbolic sin double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuC[,], Cos, tolerance=.Machine$double.eps^0.5, 
                 info="cos double matrix elements not equivalent",
                 check.attributes=FALSE)    
    expect_equal(fgpuAC[,], Acos, tolerance=.Machine$double.eps^0.5, 
                 info="arc cos double matrix elements not equivalent",
                 check.attributes=FALSE)    
    expect_equal(fgpuHC[,], Hcos, tolerance=.Machine$double.eps^0.5, 
                 info="hyperbolic cos double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuT[,], Tan, tolerance=1e-06, 
                 info="tan double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuAT[,], Atan, tolerance=.Machine$double.eps^0.5, 
                 info="arc tan double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuHT[,], Htan, tolerance=.Machine$double.eps^0.5, 
                 info="hyperbolic tan double matrix elements not equivalent",
                 check.attributes=FALSE)  
})

test_that("CPU Inplace gpuVector Single Precision Matrix Element-Wise Trignometry", {
    
    has_cpu_skip()
    
    Sin <- sin(A_vec)
    Asin <- suppressWarnings(asin(A_vec))
    Hsin <- sinh(A_vec)
    Cos <- cos(A_vec)
    Acos <- suppressWarnings(acos(A_vec))
    Hcos <- cosh(A_vec)
    Tan <- tan(A_vec) 
    Atan <- atan(A_vec)
    Htan <- tanh(A_vec)
    
    fgpuS <- gpuVector(A_vec, type="float")
    inplace(`sin`, fgpuS)
    fgpuAS <- gpuVector(A_vec, type="float")
    inplace(`asin`, fgpuAS)
    fgpuHS <- gpuVector(A_vec, type="float")
    inplace(`sinh`, fgpuHS)
    fgpuC <- gpuVector(A_vec, type="float")
    inplace(`cos`, fgpuC)
    fgpuAC <- gpuVector(A_vec, type="float")
    inplace(`acos`, fgpuAC)
    fgpuHC <- gpuVector(A_vec, type="float")
    inplace(`cosh`, fgpuHC)
    fgpuT <- gpuVector(A_vec, type="float")
    inplace(`tan`, fgpuT)
    fgpuAT <- gpuVector(A_vec, type="float")
    inplace(`atan`,fgpuAT)
    fgpuHT <- gpuVector(A_vec, type="float")
    inplace(`tanh`, fgpuHT)
    
    expect_is(fgpuC, "fgpuVector")
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

test_that("CPU Inplace gpuVector Double Precision Matrix Element-Wise Trignometry", {
    
    has_cpu_skip()
    
    Sin <- sin(A_vec)
    Asin <- suppressWarnings(asin(A_vec))
    Hsin <- sinh(A_vec)
    Cos <- cos(A_vec)
    Acos <- suppressWarnings(acos(A_vec))
    Hcos <- cosh(A_vec)
    Tan <- tan(A_vec) 
    Atan <- atan(A_vec)
    Htan <- tanh(A_vec)
    
    fgpuS <- gpuVector(A_vec, type="double")
    inplace(`sin`, fgpuS)
    fgpuAS <- gpuVector(A_vec, type="double")
    inplace(`asin`, fgpuAS)
    fgpuHS <- gpuVector(A_vec, type="double")
    inplace(`sinh`, fgpuHS)
    fgpuC <- gpuVector(A_vec, type="double")
    inplace(`cos`, fgpuC)
    fgpuAC <- gpuVector(A_vec, type="double")
    inplace(`acos`, fgpuAC)
    fgpuHC <- gpuVector(A_vec, type="double")
    inplace(`cosh`, fgpuHC)
    fgpuT <- gpuVector(A_vec, type="double")
    inplace(`tan`, fgpuT)
    fgpuAT <- gpuVector(A_vec, type="double")
    inplace(`atan`,fgpuAT)
    fgpuHT <- gpuVector(A_vec, type="double")
    inplace(`tanh`, fgpuHT)
    
    expect_equal(fgpuS[,], Sin, tolerance=.Machine$double.eps^0.5, 
                 info="sin double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuAS[,], Asin, tolerance=.Machine$double.eps^0.5, 
                 info="arc sin double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuHS[,], Hsin, tolerance=.Machine$double.eps^0.5, 
                 info="hyperbolic sin double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuC[,], Cos, tolerance=.Machine$double.eps^0.5, 
                 info="cos double matrix elements not equivalent",
                 check.attributes=FALSE)    
    expect_equal(fgpuAC[,], Acos, tolerance=.Machine$double.eps^0.5, 
                 info="arc cos double matrix elements not equivalent",
                 check.attributes=FALSE)    
    expect_equal(fgpuHC[,], Hcos, tolerance=.Machine$double.eps^0.5, 
                 info="hyperbolic cos double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuT[,], Tan, tolerance=1e-06, 
                 info="tan double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuAT[,], Atan, tolerance=.Machine$double.eps^0.5, 
                 info="arc tan double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuHT[,], Htan, tolerance=.Machine$double.eps^0.5, 
                 info="hyperbolic tan double matrix elements not equivalent",
                 check.attributes=FALSE)  
})

test_that("CPU Inplace vclMatrix Single Precision Matrix Element-Wise Trignometry", {
    
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
    
    fgpuS <- vclMatrix(A, type="float")
    inplace(`sin`, fgpuS)
    fgpuAS <- vclMatrix(A, type="float")
    inplace(`asin`, fgpuAS)
    fgpuHS <- vclMatrix(A, type="float")
    inplace(`sinh`, fgpuHS)
    fgpuC <- vclMatrix(A, type="float")
    inplace(`cos`, fgpuC)
    fgpuAC <- vclMatrix(A, type="float")
    inplace(`acos`, fgpuAC)
    fgpuHC <- vclMatrix(A, type="float")
    inplace(`cosh`, fgpuHC)
    fgpuT <- vclMatrix(A, type="float")
    inplace(`tan`, fgpuT)
    fgpuAT <- vclMatrix(A, type="float")
    inplace(`atan`,fgpuAT)
    fgpuHT <- vclMatrix(A, type="float")
    inplace(`tanh`, fgpuHT)
    
    expect_is(fgpuC, "fvclMatrix")
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

test_that("CPU Inplace vclMatrix Double Precision Matrix Element-Wise Trignometry", {
    
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
    
    fgpuS <- vclMatrix(A, type="double")
    inplace(`sin`, fgpuS)
    fgpuAS <- vclMatrix(A, type="double")
    inplace(`asin`, fgpuAS)
    fgpuHS <- vclMatrix(A, type="double")
    inplace(`sinh`, fgpuHS)
    fgpuC <- vclMatrix(A, type="double")
    inplace(`cos`, fgpuC)
    fgpuAC <- vclMatrix(A, type="double")
    inplace(`acos`, fgpuAC)
    fgpuHC <- vclMatrix(A, type="double")
    inplace(`cosh`, fgpuHC)
    fgpuT <- vclMatrix(A, type="double")
    inplace(`tan`, fgpuT)
    fgpuAT <- vclMatrix(A, type="double")
    inplace(`atan`,fgpuAT)
    fgpuHT <- vclMatrix(A, type="double")
    inplace(`tanh`, fgpuHT)
    
    expect_equal(fgpuS[,], Sin, tolerance=.Machine$double.eps^0.5, 
                 info="sin double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuAS[,], Asin, tolerance=.Machine$double.eps^0.5, 
                 info="arc sin double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuHS[,], Hsin, tolerance=.Machine$double.eps^0.5, 
                 info="hyperbolic sin double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuC[,], Cos, tolerance=.Machine$double.eps^0.5, 
                 info="cos double matrix elements not equivalent",
                 check.attributes=FALSE)    
    expect_equal(fgpuAC[,], Acos, tolerance=.Machine$double.eps^0.5, 
                 info="arc cos double matrix elements not equivalent",
                 check.attributes=FALSE)    
    expect_equal(fgpuHC[,], Hcos, tolerance=.Machine$double.eps^0.5, 
                 info="hyperbolic cos double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuT[,], Tan, tolerance=1e-06, 
                 info="tan double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuAT[,], Atan, tolerance=.Machine$double.eps^0.5, 
                 info="arc tan double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuHT[,], Htan, tolerance=.Machine$double.eps^0.5, 
                 info="hyperbolic tan double matrix elements not equivalent",
                 check.attributes=FALSE)  
})

test_that("CPU Inplace vclVector Single Precision Matrix Element-Wise Trignometry", {
    
    has_cpu_skip()
    
    Sin <- sin(A_vec)
    Asin <- suppressWarnings(asin(A_vec))
    Hsin <- sinh(A_vec)
    Cos <- cos(A_vec)
    Acos <- suppressWarnings(acos(A_vec))
    Hcos <- cosh(A_vec)
    Tan <- tan(A_vec) 
    Atan <- atan(A_vec)
    Htan <- tanh(A_vec)
    
    fgpuS <- vclVector(A_vec, type="float")
    inplace(`sin`, fgpuS)
    fgpuAS <- vclVector(A_vec, type="float")
    inplace(`asin`, fgpuAS)
    fgpuHS <- vclVector(A_vec, type="float")
    inplace(`sinh`, fgpuHS)
    fgpuC <- vclVector(A_vec, type="float")
    inplace(`cos`, fgpuC)
    fgpuAC <- vclVector(A_vec, type="float")
    inplace(`acos`, fgpuAC)
    fgpuHC <- vclVector(A_vec, type="float")
    inplace(`cosh`, fgpuHC)
    fgpuT <- vclVector(A_vec, type="float")
    inplace(`tan`, fgpuT)
    fgpuAT <- vclVector(A_vec, type="float")
    inplace(`atan`,fgpuAT)
    fgpuHT <- vclVector(A_vec, type="float")
    inplace(`tanh`, fgpuHT)
    
    expect_is(fgpuC, "fvclVector")
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

test_that("CPU Inplace vclVector Double Precision Matrix Element-Wise Trignometry", {
    
    has_cpu_skip()
    
    Sin <- sin(A_vec)
    Asin <- suppressWarnings(asin(A_vec))
    Hsin <- sinh(A_vec)
    Cos <- cos(A_vec)
    Acos <- suppressWarnings(acos(A_vec))
    Hcos <- cosh(A_vec)
    Tan <- tan(A_vec) 
    Atan <- atan(A_vec)
    Htan <- tanh(A_vec)
    
    fgpuS <- vclVector(A_vec, type="double")
    inplace(`sin`, fgpuS)
    fgpuAS <- vclVector(A_vec, type="double")
    inplace(`asin`, fgpuAS)
    fgpuHS <- vclVector(A_vec, type="double")
    inplace(`sinh`, fgpuHS)
    fgpuC <- vclVector(A_vec, type="double")
    inplace(`cos`, fgpuC)
    fgpuAC <- vclVector(A_vec, type="double")
    inplace(`acos`, fgpuAC)
    fgpuHC <- vclVector(A_vec, type="double")
    inplace(`cosh`, fgpuHC)
    fgpuT <- vclVector(A_vec, type="double")
    inplace(`tan`, fgpuT)
    fgpuAT <- vclVector(A_vec, type="double")
    inplace(`atan`,fgpuAT)
    fgpuHT <- vclVector(A_vec, type="double")
    inplace(`tanh`, fgpuHT)
    
    expect_equal(fgpuS[,], Sin, tolerance=.Machine$double.eps^0.5, 
                 info="sin double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuAS[,], Asin, tolerance=.Machine$double.eps^0.5, 
                 info="arc sin double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuHS[,], Hsin, tolerance=.Machine$double.eps^0.5, 
                 info="hyperbolic sin double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuC[,], Cos, tolerance=.Machine$double.eps^0.5, 
                 info="cos double matrix elements not equivalent",
                 check.attributes=FALSE)    
    expect_equal(fgpuAC[,], Acos, tolerance=.Machine$double.eps^0.5, 
                 info="arc cos double matrix elements not equivalent",
                 check.attributes=FALSE)    
    expect_equal(fgpuHC[,], Hcos, tolerance=.Machine$double.eps^0.5, 
                 info="hyperbolic cos double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuT[,], Tan, tolerance=1e-06, 
                 info="tan double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuAT[,], Atan, tolerance=.Machine$double.eps^0.5, 
                 info="arc tan double matrix elements not equivalent",
                 check.attributes=FALSE)  
    expect_equal(fgpuHT[,], Htan, tolerance=.Machine$double.eps^0.5, 
                 info="hyperbolic tan double matrix elements not equivalent",
                 check.attributes=FALSE)  
})

test_that("CPU Inplace gpuMatrix Single Precision Matrix Exponential", {
    has_cpu_skip()
    
    R_exp <- exp(A)
    
    fgpuA <- gpuMatrix(A, type="float")
    
    inplace(`exp`,fgpuA)
    
    expect_is(fgpuA, "fgpuMatrix")
    expect_equal(fgpuA[,], R_exp, tolerance=1e-07, 
                 info="exp float matrix elements not equivalent",
                 check.attributes=FALSE)   
})

test_that("CPU Inplace gpuMatrix Double Precision Matrix Exponential", {
    has_cpu_skip()
    
    R_exp <- exp(A)
    
    fgpuA <- gpuMatrix(A, type="double")
    
    inplace(`exp`,fgpuA)
    
    expect_is(fgpuA, "dgpuMatrix")
    expect_equal(fgpuA[,], R_exp, tolerance=.Machine$double.eps^0.5, 
                 info="exp double matrix elements not equivalent",
                 check.attributes=FALSE)   
})

test_that("CPU Inplace gpuVector Single Precision Matrix Exponential", {
    has_cpu_skip()
    
    R_exp <- exp(A_vec)
    
    fgpuA <- gpuVector(A_vec, type="float")
    
    inplace(`exp`,fgpuA)
    
    expect_is(fgpuA, "fgpuVector")
    expect_equal(fgpuA[,], R_exp, tolerance=1e-07, 
                 info="exp float vector elements not equivalent",
                 check.attributes=FALSE)   
})

test_that("CPU Inplace gpuVector Double Precision Matrix Exponential", {
    has_cpu_skip()
    
    R_exp <- exp(A_vec)
    
    fgpuA <- gpuVector(A_vec, type="double")
    
    inplace(`exp`,fgpuA)
    
    expect_is(fgpuA, "dgpuVector")
    expect_equal(fgpuA[,], R_exp, tolerance=.Machine$double.eps^0.5, 
                 info="exp double vector elements not equivalent",
                 check.attributes=FALSE)   
})

test_that("CPU Inplace vclMatrix Single Precision Matrix Exponential", {
    has_cpu_skip()
    
    R_exp <- exp(A)
    
    fgpuA <- vclMatrix(A, type="float")
    
    inplace(`exp`,fgpuA)
    
    expect_is(fgpuA, "fvclMatrix")
    expect_equal(fgpuA[,], R_exp, tolerance=1e-07, 
                 info="exp float matrix elements not equivalent",
                 check.attributes=FALSE)   
})

test_that("CPU Inplace vclMatrix Double Precision Matrix Exponential", {
    has_cpu_skip()
    
    R_exp <- exp(A)
    
    fgpuA <- vclMatrix(A, type="double")
    
    inplace(`exp`,fgpuA)
    
    expect_is(fgpuA, "dvclMatrix")
    expect_equal(fgpuA[,], R_exp, tolerance=.Machine$double.eps^0.5, 
                 info="exp double matrix elements not equivalent",
                 check.attributes=FALSE)   
})

test_that("CPU Inplace vclVector Single Precision Matrix Exponential", {
    has_cpu_skip()
    
    R_exp <- exp(A_vec)
    
    fgpuA <- vclVector(A_vec, type="float")
    
    inplace(`exp`,fgpuA)
    
    expect_is(fgpuA, "fvclVector")
    expect_equal(fgpuA[,], R_exp, tolerance=1e-07, 
                 info="exp float vector elements not equivalent",
                 check.attributes=FALSE)   
})

test_that("CPU Inplace vclVector Double Precision Matrix Exponential", {
    has_cpu_skip()
    
    R_exp <- exp(A_vec)
    
    fgpuA <- vclVector(A_vec, type="double")
    
    inplace(`exp`,fgpuA)
    
    expect_is(fgpuA, "dvclVector")
    expect_equal(fgpuA[,], R_exp, tolerance=.Machine$double.eps^0.5, 
                 info="exp double vector elements not equivalent",
                 check.attributes=FALSE)   
})

test_that("CPU Inplace gpuMatrix Single Precision Matrix Absolute Value", {
    has_cpu_skip()
    
    R_abs <- abs(A)
    
    fgpuA <- gpuMatrix(A, type="float")
    
    inplace(`abs`,fgpuA)
    
    expect_is(fgpuA, "fgpuMatrix")
    expect_equal(fgpuA[,], R_abs, tolerance=1e-07, 
                 info="abs float matrix elements not equivalent",
                 check.attributes=FALSE)   
})

test_that("CPU Inplace gpuMatrix Double Precision Matrix Absolute Value", {
    has_cpu_skip()
    
    R_abs <- abs(A)
    
    fgpuA <- gpuMatrix(A, type="double")
    
    inplace(`abs`,fgpuA)
    
    expect_is(fgpuA, "dgpuMatrix")
    expect_equal(fgpuA[,], R_abs, tolerance=.Machine$double.eps^0.5, 
                 info="abs double matrix elements not equivalent",
                 check.attributes=FALSE)   
})

test_that("CPU Inplace gpuVector Single Precision Matrix Absolute Value", {
    has_cpu_skip()
    
    R_abs <- abs(A_vec)
    
    fgpuA <- gpuVector(A_vec, type="float")
    
    inplace(`abs`,fgpuA)
    
    expect_is(fgpuA, "fgpuVector")
    expect_equal(fgpuA[,], R_abs, tolerance=1e-07, 
                 info="abs float vector elements not equivalent",
                 check.attributes=FALSE)   
})

test_that("CPU Inplace gpuVector Double Precision Matrix Absolute Value", {
    has_cpu_skip()
    
    R_abs <- abs(A_vec)
    
    fgpuA <- gpuVector(A_vec, type="double")
    
    inplace(`abs`,fgpuA)
    
    expect_is(fgpuA, "dgpuVector")
    expect_equal(fgpuA[,], R_abs, tolerance=.Machine$double.eps^0.5, 
                 info="abs double vector elements not equivalent",
                 check.attributes=FALSE)   
})

test_that("CPU Inplace vclMatrix Single Precision Matrix Absolute Value", {
    has_cpu_skip()
    
    R_abs <- abs(A)
    
    fgpuA <- vclMatrix(A, type="float")
    
    inplace(`abs`,fgpuA)
    
    expect_is(fgpuA, "fvclMatrix")
    expect_equal(fgpuA[,], R_abs, tolerance=1e-07, 
                 info="abs float matrix elements not equivalent",
                 check.attributes=FALSE)   
})

test_that("CPU Inplace vclMatrix Double Precision Matrix Absolute Value", {
    has_cpu_skip()
    
    R_abs <- abs(A)
    
    fgpuA <- vclMatrix(A, type="double")
    
    inplace(`abs`,fgpuA)
    
    expect_is(fgpuA, "dvclMatrix")
    expect_equal(fgpuA[,], R_abs, tolerance=.Machine$double.eps^0.5, 
                 info="abs double matrix elements not equivalent",
                 check.attributes=FALSE)   
})

test_that("CPU Inplace vclVector Single Precision Matrix Absolute Value", {
    has_cpu_skip()
    
    R_abs <- abs(A_vec)
    
    fgpuA <- vclVector(A_vec, type="float")
    
    inplace(`abs`,fgpuA)
    
    expect_is(fgpuA, "fvclVector")
    expect_equal(fgpuA[,], R_abs, tolerance=1e-07, 
                 info="abs float vector elements not equivalent",
                 check.attributes=FALSE)   
})

test_that("CPU Inplace vclVector Double Precision Matrix Absolute Value", {
    has_cpu_skip()
    
    R_abs <- abs(A_vec)
    
    fgpuA <- vclVector(A_vec, type="double")
    
    inplace(`abs`,fgpuA)
    
    expect_is(fgpuA, "dvclVector")
    expect_equal(fgpuA[,], R_abs, tolerance=.Machine$double.eps^0.5, 
                 info="abs double vector elements not equivalent",
                 check.attributes=FALSE)   
})

options(warn=0)

setContext(current_context)



