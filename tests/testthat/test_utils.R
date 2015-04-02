library(gpuR)
context("Utility Functions")

test_that("detectDevices accepts appropriate input", {
    expect_error(detectGPUs(0))
    expect_error(detectGPUs(2))
    expect_error(detectGPUs(c(2,3)))
})
