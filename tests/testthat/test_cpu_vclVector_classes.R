library(gpuR)
context("CPU vclVector classes")

# set option to use CPU instead of GPU
options(gpuR.default.device = "cpu")

set.seed(123)
A <- seq.int(10)
D <- rnorm(10)

test_that("vclVector integer class initializer" ,{
    
    vclA <- vclVector(A)
    
    expect_is(vclA, "ivclVector")
    expect_equivalent(vclA[], A, 
                      info="vcl integer vector elements not equivalent")
    expect_equal(length(vclA), length(A))
    expect_equal(typeof(vclA), "integer")
})

test_that("vclVector float class initializer" ,{
    
    vclD <- vclVector(D, type="float")
    
    expect_is(vclD, "fvclVector")
    expect_equal(vclD[], D, tolerance=1e-07, 
                 info="vcl float vector elements not equivalent")
    expect_equal(length(vclD), length(D))
    expect_equal(typeof(vclD), "float")
})

test_that("vclVector double class initializer" ,{
    
    vclD <- vclVector(D)
    
    expect_is(vclD, "dvclVector")
    expect_equal(vclD[], D, tolerance=.Machine$double.eps ^ 0.5, 
                 info="vcl double vector elements not equivalent")
    expect_equal(length(vclD), length(D))
    expect_equal(typeof(vclD), "double")
})

options(gpuR.default.device = "gpu")

