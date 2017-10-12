library(gpuR)
context("CPU Inplace Algebra Operations")

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
scalar <- 2

# Addition
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

test_that("CPU inplace gpuMatrix-scalar Addition", {
    
    has_cpu_skip()
    
    C <- A + scalar
    
    fgpuA <- gpuMatrix(A, type="float")
    
    inplace(`+`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- gpuMatrix(A, type="double")
    
    inplace(`+`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace scalar-gpuMatrix Addition", {
    
    has_cpu_skip()
    
    C <- scalar + B
    
    fgpuB <- gpuMatrix(B, type="float")
    
    inplace(`+`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuB <- gpuMatrix(B, type="double")
    
    inplace(`+`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=.Machine$double.eps^0.5, 
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

test_that("CPU inplace vclMatrix-scalar Addition", {
    
    has_cpu_skip()
    
    C <- A + scalar
    
    fgpuA <- vclMatrix(A, type="float")
    
    inplace(`+`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- vclMatrix(A, type="double")
    
    inplace(`+`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace scalar-vclMatrix Addition", {
    
    has_cpu_skip()
    
    C <- scalar + B
    
    fgpuB <- vclMatrix(B, type="float")
    
    inplace(`+`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuB <- vclMatrix(B, type="double")
    
    inplace(`+`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=.Machine$double.eps^0.5, 
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

test_that("CPU inplace gpuVector-scalar Addition", {
    
    has_cpu_skip()
    
    C <- A_vec + scalar
    
    fgpuA <- gpuVector(A, type="float")
    
    inplace(`+`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- gpuVector(A, type="double")
    
    inplace(`+`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace scalar-gpuVector Addition", {
    
    has_cpu_skip()
    
    C <- scalar + B_vec
    
    fgpuB <- gpuVector(B, type="float")
    
    inplace(`+`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuB <- gpuVector(B, type="double")
    
    inplace(`+`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=.Machine$double.eps^0.5, 
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

test_that("CPU inplace vclVector-scalar Addition", {
    
    has_cpu_skip()
    
    C <- A_vec + scalar
    
    fgpuA <- vclVector(A, type="float")
    
    inplace(`+`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- vclVector(A, type="double")
    
    inplace(`+`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace scalar-vclVector Addition", {
    
    has_cpu_skip()
    
    C <- scalar + B_vec
    
    fgpuB <- vclVector(B, type="float")
    
    inplace(`+`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuB <- vclVector(B, type="double")
    
    inplace(`+`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

# subtraction
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

test_that("CPU inplace gpuMatrix-scalar Subtraction", {
    
    has_cpu_skip()
    
    C <- A - scalar
    
    fgpuA <- gpuMatrix(A, type="float")
    
    inplace(`-`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- gpuMatrix(A, type="double")
    
    inplace(`-`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace scalar-gpuMatrix Subtraction", {
    
    has_cpu_skip()
    
    C <- scalar - B
    
    fgpuB <- gpuMatrix(B, type="float")
    
    inplace(`-`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuB <- gpuMatrix(B, type="double")
    
    inplace(`-`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=.Machine$double.eps^0.5, 
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

test_that("CPU inplace vclMatrix-scalar Subtraction", {
    
    has_cpu_skip()
    
    C <- A - scalar
    
    fgpuA <- vclMatrix(A, type="float")
    
    inplace(`-`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- vclMatrix(A, type="double")
    
    inplace(`-`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace scalar-vclMatrix Subtraction", {
    
    has_cpu_skip()
    
    C <- scalar - B
    
    fgpuB <- vclMatrix(B, type="float")
    
    inplace(`-`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuB <- vclMatrix(B, type="double")
    
    inplace(`-`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=.Machine$double.eps^0.5, 
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

test_that("CPU inplace gpuVector-scalar Subtraction", {
    
    has_cpu_skip()
    
    C <- A_vec - scalar
    
    fgpuA <- gpuVector(A, type="float")
    
    inplace(`-`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- gpuVector(A, type="double")
    
    inplace(`-`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace scalar-gpuVector Subraction", {
    
    has_cpu_skip()
    
    C <- scalar - B_vec
    
    fgpuB <- gpuVector(B, type="float")
    
    inplace(`-`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuB <- gpuVector(B, type="double")
    
    inplace(`-`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=.Machine$double.eps^0.5, 
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

test_that("CPU inplace vclVector-scalar Subtraction", {
    
    has_cpu_skip()
    
    C <- A_vec - scalar
    
    fgpuA <- vclVector(A, type="float")
    
    inplace(`-`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- vclVector(A, type="double")
    
    inplace(`-`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace scalar-vclVector Subraction", {
    
    has_cpu_skip()
    
    C <- scalar - B_vec
    
    fgpuB <- vclVector(B, type="float")
    
    inplace(`-`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuB <- vclVector(B, type="double")
    
    inplace(`-`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

# elementwise multiplication
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

test_that("CPU inplace gpuMatrix-scalar Elementwise Multiplication", {
    
    has_cpu_skip()
    
    C <- A * scalar
    
    fgpuA <- gpuMatrix(A, type="float")
    
    inplace(`*`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- gpuMatrix(A, type="double")
    
    inplace(`*`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace scalar-gpuMatrix Elementwise Multiplication", {
    
    has_cpu_skip()
    
    C <- scalar * B
    
    fgpuB <- gpuMatrix(B, type="float")
    
    inplace(`*`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuB <- gpuMatrix(B, type="double")
    
    inplace(`*`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=.Machine$double.eps^0.5, 
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

test_that("CPU inplace vclMatrix-scalar Elementwise Multiplication", {
    
    has_cpu_skip()
    
    C <- A * scalar
    
    fgpuA <- vclMatrix(A, type="float")
    
    inplace(`*`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- vclMatrix(A, type="double")
    
    inplace(`*`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace scalar-vclMatrix Elementwise Multiplication", {
    
    has_cpu_skip()
    
    C <- scalar * B
    
    fgpuB <- vclMatrix(B, type="float")
    
    inplace(`*`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuB <- vclMatrix(B, type="double")
    
    inplace(`*`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace gpuVector Elementwise Multiplication", {

    has_cpu_skip()

    C <- A_vec * B_vec

    fgpuA <- gpuVector(A_vec, type="float")
    fgpuB <- gpuVector(B_vec, type="float")

    inplace(`*`, fgpuA, fgpuB)

    expect_equal(fgpuA[,], C, tolerance=1e-07,
                 info="float matrix elements not equivalent")

    fgpuA <- gpuVector(A_vec, type="double")
    fgpuB <- gpuVector(B_vec, type="double")

    inplace(`*`, fgpuA, fgpuB)

    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5,
                 info="double matrix elements not equivalent")

})

test_that("CPU inplace gpuVector-scalar Elementwise Multiplication", {

    has_cpu_skip()

    C <- A_vec * scalar

    fgpuA <- gpuVector(A_vec, type="float")

    inplace(`*`, fgpuA, scalar)

    expect_equal(fgpuA[,], C, tolerance=1e-07,
                 info="float matrix elements not equivalent")

    fgpuA <- gpuVector(A_vec, type="double")

    inplace(`*`, fgpuA, scalar)

    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5,
                 info="double matrix elements not equivalent")

})

test_that("CPU inplace scalar-gpuVector Elementwise Multiplication", {
    
    has_cpu_skip()
    
    C <- scalar * B_vec
    
    fgpuB <- gpuVector(B_vec, type="float")

    inplace(`*`, scalar, fgpuB)

    expect_equal(fgpuB[,], C, tolerance=1e-07,
                 info="float matrix elements not equivalent")
    
    fgpuB <- gpuVector(B_vec, type="double")

    inplace(`*`, scalar, fgpuB)

    expect_equal(fgpuB[,], C, tolerance=.Machine$double.eps^0.5,
                 info="double matrix elements not equivalent")
    
})

test_that("CPU inplace vclVector Elementwise Multiplication", {
    
    has_cpu_skip()
    
    C <- A_vec * B_vec
    
    fgpuA <- vclVector(A_vec, type="float")
    fgpuB <- vclVector(B_vec, type="float")
    
    inplace(`*`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- vclVector(A_vec, type="double")
    fgpuB <- vclVector(B_vec, type="double")
    
    inplace(`*`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace vclVector-scalar Elementwise Multiplication", {
    
    has_cpu_skip()
    
    C <- A_vec * scalar
    
    fgpuA <- vclVector(A_vec, type="float")
    
    inplace(`*`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- vclVector(A_vec, type="double")
    
    inplace(`*`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace scalar-vclVector Elementwise Multiplication", {
    
    has_cpu_skip()
    
    C <- scalar * B_vec
    
    fgpuB <- vclVector(B_vec, type="float")
    
    inplace(`*`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuB <- vclVector(B_vec, type="double")
    
    inplace(`*`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

# elementwise division
test_that("CPU inplace gpuMatrix Elementwise Division", {
    
    has_cpu_skip()
    
    C <- A / B
    
    fgpuA <- gpuMatrix(A, type="float")
    fgpuB <- gpuMatrix(B, type="float")
    
    inplace(`/`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- gpuMatrix(A, type="double")
    fgpuB <- gpuMatrix(B, type="double")
    
    inplace(`/`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace gpuMatrix-scalar Elementwise Division", {
    
    has_cpu_skip()
    
    C <- A / scalar
    
    fgpuA <- gpuMatrix(A, type="float")
    
    inplace(`/`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- gpuMatrix(A, type="double")
    
    inplace(`/`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace scalar-gpuMatrix Elementwise Division", {
    
    has_cpu_skip()
    
    C <- scalar / B
    
    fgpuB <- gpuMatrix(B, type="float")
    
    inplace(`/`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuB <- gpuMatrix(B, type="double")
    
    inplace(`/`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace vclMatrix Elementwise Division", {
    
    has_cpu_skip()
    
    C <- A / B
    
    fgpuA <- vclMatrix(A, type="float")
    fgpuB <- vclMatrix(B, type="float")
    
    inplace(`/`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- gpuMatrix(A, type="double")
    fgpuB <- gpuMatrix(B, type="double")
    
    inplace(`/`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace vclMatrix-scalar Elementwise Division", {
    
    has_cpu_skip()
    
    C <- A / scalar
    
    fgpuA <- vclMatrix(A, type="float")
    
    inplace(`/`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- vclMatrix(A, type="double")
    
    inplace(`/`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace scalar-vclMatrix Elementwise Division", {
    
    has_cpu_skip()
    
    C <- scalar / B
    
    fgpuB <- vclMatrix(B, type="float")
    
    inplace(`/`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuB <- vclMatrix(B, type="double")
    
    inplace(`/`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace gpuVector Elementwise Division", {
    
    has_cpu_skip()
    
    C <- A_vec / B_vec
    
    fgpuA <- gpuVector(A, type="float")
    fgpuB <- gpuVector(B, type="float")
    
    inplace(`/`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- gpuVector(A, type="double")
    fgpuB <- gpuVector(B, type="double")
    
    inplace(`/`, fgpuA, fgpuB)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace gpuVector-scalar Elementwise Division", {
    
    has_cpu_skip()
    
    C <- A_vec / scalar
    
    fgpuA <- gpuVector(A, type="float")
    
    inplace(`/`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuA <- gpuVector(A, type="double")
    
    inplace(`/`, fgpuA, scalar)
    
    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace scalar-gpuVector Elementwise Division", {
    
    has_cpu_skip()
    
    C <- scalar / B_vec
    
    fgpuB <- gpuVector(B, type="float")
    
    inplace(`/`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuB <- gpuVector(B, type="double")
    
    inplace(`/`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

test_that("CPU inplace vclVector Elementwise Division", {

    has_cpu_skip()

    C <- A_vec / B_vec

    fgpuA <- vclVector(A, type="float")
    fgpuB <- vclVector(B, type="float")

    inplace(`/`, fgpuA, fgpuB)

    expect_equal(fgpuA[,], C, tolerance=1e-07,
                 info="float matrix elements not equivalent")

    fgpuA <- vclVector(A, type="double")
    fgpuB <- vclVector(B, type="double")

    inplace(`/`, fgpuA, fgpuB)

    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5,
                 info="double matrix elements not equivalent")

})

test_that("CPU inplace vclVector-scalar Elementwise Division", {

    has_cpu_skip()

    C <- A_vec / scalar

    fgpuA <- vclVector(A, type="float")

    inplace(`/`, fgpuA, scalar)

    expect_equal(fgpuA[,], C, tolerance=1e-07,
                 info="float matrix elements not equivalent")

    fgpuA <- vclVector(A, type="double")

    inplace(`/`, fgpuA, scalar)

    expect_equal(fgpuA[,], C, tolerance=.Machine$double.eps^0.5,
                 info="double matrix elements not equivalent")

})

test_that("CPU inplace scalar-vclVector Elementwise Division", {
    
    has_cpu_skip()
    
    C <- scalar / B_vec
    
    fgpuB <- vclVector(B, type="float")
    
    inplace(`/`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=1e-07, 
                 info="float matrix elements not equivalent")  
    
    fgpuB <- vclVector(B, type="double")
    
    inplace(`/`, scalar, fgpuB)
    
    expect_equal(fgpuB[,], C, tolerance=.Machine$double.eps^0.5, 
                 info="double matrix elements not equivalent")  
    
})

setContext(current_context)
