library(bigGPU)
context("vector addition")

A <- seq.int(from=0, to=999)
B <- seq.int(from=1000, to=1)

C <- A + B


# test_that("can read cl file", {
#     print(test())
# })

test_that("vector additonal successful", {
    gpuC <- gpu_vec_add(A, B)

    expect_equivalent(gpuC, C)
})
