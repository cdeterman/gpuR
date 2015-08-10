library(gpuR)
context("vclMatrix classes")

set.seed(123)
A <- matrix(seq.int(100), nrow=5)
D <- matrix(rnorm(100), nrow=5)


test_that("vclMatrix integer class initializer" ,{
    
    has_gpu_skip()
    
    vclA <- vclMatrix(A)
    
    expect_is(vclA, "ivclMatrix")
    expect_equivalent(vclA[], A, 
                      info="vcl integer matrix elements not equivalent")
    expect_equal(dim(vclA), dim(A))
    expect_equal(ncol(vclA), ncol(A))
    expect_equal(nrow(vclA), nrow(A))
    expect_equal(typeof(vclA), "integer")
})

test_that("vclMatrix float class initializer" ,{
    
    has_gpu_skip()
    
    vclD <- vclMatrix(D, type="float")
    
    expect_is(vclD, "fvclMatrix")
    expect_equal(vclD[], D, tolerance=1e-07, 
                      info="vcl float matrix elements not equivalent")
    expect_equal(dim(vclD), dim(D))
    expect_equal(ncol(vclD), ncol(D))
    expect_equal(nrow(vclD), nrow(D))
    expect_equal(typeof(vclD), "float")
})

test_that("vclMatrix double class initializer" ,{
    
    has_gpu_skip()
    
    vclD <- vclMatrix(D)
    
    expect_is(vclD, "dvclMatrix")
    expect_equal(vclD[], D, tolerance=.Machine$double.eps ^ 0.5, 
                      info="vcl double matrix elements not equivalent")
    expect_equal(dim(vclD), dim(D))
    expect_equal(ncol(vclD), ncol(D))
    expect_equal(nrow(vclD), nrow(D))
    expect_equal(typeof(vclD), "double")
})
