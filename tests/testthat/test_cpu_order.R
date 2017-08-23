library(gpuR)
context("CPU Ordering Methods")

current_context <- set_device_context("cpu")

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
A <- matrix(rnorm(ORDER^2), nrow=ORDER)
idx <- sample(seq.int(ORDER), ORDER)

C <- A[idx,]

test_that("CPU vclMatrix permute",
          {
              has_cpu_skip()
              
              fgpuX <- vclMatrix(A, type="float")
              
              gpuC <- permute(fgpuX, order = idx)
              
              expect_is(gpuC, "fvclMatrix")
              expect_equal(gpuC[], C, tolerance=1e-06, 
                           info="float row permutations not equivalent")  
              
              fgpuX <- vclMatrix(A, type="double")
              
              gpuC <- permute(fgpuX, order = idx)
              
              expect_is(gpuC, "dvclMatrix")
              expect_equal(gpuC[], C, tolerance=.Machine$double.eps^0.5, 
                           info="double row permutations not equivalent")  
          })

setContext(current_context)
