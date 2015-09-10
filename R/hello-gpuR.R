#' @title GPU functions for R Objects
#' 
#' @description This package was developed to provide simple to use R functions
#' that leverage the power of GPU's but also retain a format familiar to the
#' R user.  There are a handfull of other R packages that provide some
#' GPU functionality but nearly all rely on a CUDA backend thereby restricting
#' the user to NVIDIA GPU hardware.  In the spirit of being as broadly 
#' applicable as possible, this GPU code herein relies upon OpenCL via the 
#' ViennaCL library.  
#' 
#' OpenCL, in contrast to CUDA, is open source and can be used across 
#' different graphics cards (e.g. NVIDIA, AMD, Intel).  This package removes 
#' the complex code needed for GPU computing and provides easier to use 
#' functions to apply on R objects.
#' 
#' \tabular{ll}{ Package: \tab gpuR\cr Type: \tab Package\cr 
#' Version: \tab 1.0.0\cr Date: \tab 2015-03-31\cr License: \tab GPL-3\cr
#' Copyright: \tab (c) 2015 Charles E. Determan Jr.\cr URL: \tab 
#' \url{http://www.github.com/cdeterman/gpuR}\cr LazyLoad: \tab yes\cr
#' }
#' 
#' 
#' @note There are other packages that also provide wrappers for OpenCL 
#' programming including \pkg{OpenCL} by Simon Urbanek and \bold{ROpenCL} at 
#' Open Analytics by Willem Ligtenberg.  Both of these packages provide
#' the R user an interface to directly call OpenCL functions.  This package, 
#' however, hides these functions so the user does not require any knowledge
#' of OpenCL to begin using their GPU.  The idea behind this package is to
#' provide a means to begin using existing algorithms without the need
#' to write extensive amounts of C/C++/OpenCL code.
#' @author 
#' Charles Determan \email{cdetermanjr@@gmail.com}
#' 
#' Maintainer: Charles Determan \email{cdetermanjr@@gmail.com}
#' @docType package
#' @name gpuR-package
#' @aliases gpuR-package gpuR
NULL