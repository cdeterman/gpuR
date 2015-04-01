library(bigGPU)
context("vector addition")

A <- seq.int(from=0, to=999)
B <- seq.int(from=1000, to=1)

C <- A + B


# test_that("can read cl file", {
#     print(test())
# })

test_that("vector additonal successful", {
    gpuA <- as.gpuVector(A)
    gpuB <- as.gpuVector(B)
    
    # manual call
    gpuC <- gpu_vec_add(A, B)
    
    # generic call
    gpuC2 <- gpuA + gpuB

    expect_equivalent(gpuC, C)
    expect_equivalent(gpuC2, C)
})
