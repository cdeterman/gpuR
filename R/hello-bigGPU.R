#' @title GPU functions for R Objects
#' 
#' @description This package was developed to provide simple to use R functions
#' that leverage the power of GPU's but also retain a format familiar to the
#' R user.  In the spirit of being as broadly applicable as possible, this
#' GPU code herein relies upon OpenCL.  OpenCL, in contrast to CUDA, is
#' open source and can be used across different graphics cards (e.g. NVIDIA,
#' AMD, Intel).  This package removes the complex code needed for GPU 
#' computing and provides easier to use functions to apply on R objects.
#' 
#' \tabular{ll}{ Package: \tab bigGPU\cr Type: \tab Package\cr 
#' Version: \tab 1.0.0\cr Date: \tab 2015-03-31\cr License: \tab GPL-3\cr
#' Copyright: \tab (c) 2015 Charles E. Determan Jr.\cr URL: \tab 
#' \url{http://www.github.com/cdeterman/bigGPU}\cr LazyLoad: \tab yes\cr
#' }
#' 
#' 
#' @note There are other packages that also provide wrappers for OpenCL 
#' programming including \pkg{OpenCL} by Simon Urbanek and \bold{ROpenCL} at 
#' Open Analytics by Willem Ligtenberg.  Both of these packages provide
#' the R user an interface to directly call OpenCL functions.  This package, 
#' however, utilizes the C++ API directly in the source code.  This decision
#' has been made for the following reasons: \preformatted{
#' 
#' 1. Any code the user writes with R wrappers will be similar to writing 
#' direct C/C++ code.
#' 
#' 2. The development of the C++ API for OpenCL is continually being developed
#' and there would likely be a delay before appropriate wrappers became 
#' existant for R users.   It would be easier to make modifications to 
#' supported C++ changes in contrast to making sure R 'plays nice' with
#' new features.
#' 
#' 3. This avoids any potential overhead caused by creating wrappers for
#' individual functions.  This point has not been benchmarked but there is 
#' always a cost and keeping all the C++ code together is preferred IMHO.
#' 
#' }
#' @author 
#' Charles Determan \email{cdetermanjr@@gmail.com}
#' 
#' Maintainer: Charles Determan \email{cdetermanjr@@gmail.com}
#' @docType package
#' @name bigGPU-package
#' @aliases bigGPU-package bigGPU
NULL