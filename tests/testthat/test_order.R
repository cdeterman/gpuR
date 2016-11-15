library(gpuR)
context("Ordering Methods")

# set seed
set.seed(123)

ORDER <- 4

# Base R objects
A <- matrix(rnorm(ORDER^2), nrow=ORDER)
idx <- sample(seq.int(ORDER), ORDER)

C <- A[idx,]

test_that("vclMatrix permute",
{
  has_gpu_skip()
  
  fgpuX <- vclMatrix(A, type="float")
  
  gpuC <- permute(fgpuX, order = idx)
  
  expect_is(gpuC, "fvclMatrix")
  expect_equal(gpuC[], C, tolerance=1e-06, 
               info="float row permutations not equivalent")  
  
  has_double_skip()
  
  fgpuX <- vclMatrix(A, type="double")
  
  gpuC <- permute(fgpuX, order = idx)
  
  expect_is(gpuC, "dvclMatrix")
  expect_equal(gpuC[], C, tolerance=.Machine$double.eps^0.5, 
               info="double row permutations not equivalent")  
})
