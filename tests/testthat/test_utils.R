library(gpuR)
context("Utility Functions")

test_that("detectGPUs() accepts appropriate input", {
    
    # should function regardless of GPUs
    expect_gte(detectGPUs(NULL), 0)
    expect_gte(detectGPUs(1L), 0)
    
    has_gpu_skip()
    
    expect_error(detectGPUs(0))
    expect_error(detectGPUs(2))
    expect_error(detectGPUs(c(2,3)))
})

test_that("gpuInfo() accepts appropriate input", {
    
    has_gpu_skip()
    
    expect_error(gpuInfo(0, 0))
    expect_error(gpuInfo(2, 2))
    expect_error(gpuInfo(0, 1))
    expect_error(gpuInfo(1, 0))
    expect_error(gpuInfo(c(2,3), 1))
    expect_error(gpuInfo(1, detectGPUs() + 1))
    
    expect_is(gpuInfo(), "list")
})

test_that("detectCPUs() accepts correct input", {
    
    # should function regardless of CPUs
    expect_gte(detectCPUs(NULL), 0)
    expect_gte(detectCPUs(1L), 0)
    
    has_cpu_skip()
    
    expect_error(detectCPUs(0))
    expect_error(detectCPUs(2))
    expect_error(detectCPUs(c(2,3)))
})

test_that("contexts behave correctly", {
    expect_error(setContext(100), 
                 info = "should through error on out-of-bounds context")
})

