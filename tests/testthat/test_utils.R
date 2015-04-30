library(gpuR)
context("Utility Functions")

check_for_gpu <- function() {
    if (detectGPUs() == 0) {
        skip("No GPUs available")
    }
}

test_that("detectGPUs() accepts appropriate input", {
    expect_error(detectGPUs(0))
    expect_error(detectGPUs(2))
    expect_error(detectGPUs(c(2,3)))
})

test_that("gpuInfo() accepts appropriate input", {
    expect_error(gpuInfo(0, 0))
    expect_error(gpuInfo(2, 2))
    expect_error(gpuInfo(0, 1))
    expect_error(gpuInfo(1, 0))
    expect_error(gpuInfo(c(2,3), 1))
    
    check_for_gpu()
    
    expect_is(gpuInfo(), "list")
})
