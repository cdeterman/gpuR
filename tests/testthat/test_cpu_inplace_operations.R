library(gpuR)
context("CPU Inplace Operations")

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


test_that("CPU inplace gpuMatrix Addition", {
    
    has_cpu_skip()
    
    C <- A + B
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    
    inplace(`+`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- gpuMatrix(A, type="double")
    fgpuB <- gpuMatrix(B, type="double")
    
    inplace(`+`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace vclMatrix Addition", {
    
    has_cpu_skip()
    
    C <- A + B
    
    fgpuA <- vclMatrix(A, type="float")
    fgpuB <- vclMatrix(B, type="float")
    
    inplace(`+`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- gpuMatrix(A, type="double")
    fgpuB <- gpuMatrix(B, type="double")
    
    inplace(`+`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace gpuVector Addition", {
    
    has_cpu_skip()
    
    C <- A_vec + B_vec
    
    fgpuA <- gpuVector(A, type="float")
    fgpuB <- gpuVector(B, type="float")
    
    inplace(`+`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- gpuVector(A, type="double")
    fgpuB <- gpuVector(B, type="double")
    
    inplace(`+`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace vclVector Addition", {
    
    has_cpu_skip()
    
    C <- A_vec + B_vec
    
    fgpuA <- vclVector(A, type="float")
    fgpuB <- vclVector(B, type="float")
    
    inplace(`+`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- vclVector(A, type="double")
    fgpuB <- vclVector(B, type="double")
    
    inplace(`+`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})


test_that("CPU inplace gpuMatrix Subtraction", {
    
    has_cpu_skip()
    
    C <- A - B
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    
    inplace(`-`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")
    
    fgpuA <- gpuMatrix(A, type="double")
    fgpuB <- gpuMatrix(B, type="double")
    
    inplace(`-`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace vclMatrix Subtraction", {
    
    has_cpu_skip()
    
    C <- A - B
    
    fgpuA <- vclMatrix(A, type="float")
    fgpuB <- vclMatrix(B, type="float")
    
    inplace(`-`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- gpuMatrix(A, type="double")
    fgpuB <- gpuMatrix(B, type="double")
    
    inplace(`-`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace gpuVector Subtraction", {
    
    has_cpu_skip()
    
    C <- A_vec - B_vec
    
    fgpuA <- gpuVector(A, type="float")
    fgpuB <- gpuVector(B, type="float")
    
    inplace(`-`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- gpuVector(A, type="double")
    fgpuB <- gpuVector(B, type="double")
    
    inplace(`-`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace vclVector Subtraction", {
    
    has_cpu_skip()
    
    C <- A_vec - B_vec
    
    fgpuA <- vclVector(A, type="float")
    fgpuB <- vclVector(B, type="float")
    
    inplace(`-`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- vclVector(A, type="double")
    fgpuB <- vclVector(B, type="double")
    
    inplace(`-`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})


test_that("CPU inplace gpuMatrix Elementwise Multiplication", {
    
    has_cpu_skip()
    
    C <- A * B
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    
    inplace(`*`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- gpuMatrix(A, type="double")
    fgpuB <- gpuMatrix(B, type="double")
    
    inplace(`*`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace vclMatrix Elementwise Multiplication", {
    
    has_cpu_skip()
    
    C <- A * B
    
    fgpuA <- vclMatrix(A, type="float")
    fgpuB <- vclMatrix(B, type="float")
    
    inplace(`*`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- gpuMatrix(A, type="double")
    fgpuB <- gpuMatrix(B, type="double")
    
    inplace(`*`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace gpuVector Elementwise Multiplication", {
    
    has_cpu_skip()
    
    C <- A_vec * B_vec
    
    fgpuA <- gpuVector(A, type="float")
    fgpuB <- gpuVector(B, type="float")
    
    inplace(`*`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- gpuVector(A, type="double")
    fgpuB <- gpuVector(B, type="double")
    
    inplace(`*`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace vclVector Elementwise Multiplication", {
    
    has_cpu_skip()
    
    C <- A_vec * B_vec
    
    fgpuA <- vclVector(A, type="float")
    fgpuB <- vclVector(B, type="float")
    
    inplace(`*`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- vclVector(A, type="double")
    fgpuB <- vclVector(B, type="double")
    
    inplace(`*`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})


setContext(current_context)
