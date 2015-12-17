
#' @title Get gpuVectorSlice type
#' @param x A gpuVectorSlice object
# @rdname typeof-methods
#' @aliases typeof,gpuVectorSlice
#' @export
setMethod('typeof', signature(x="gpuVectorSlice"),
          function(x) {
              type <- switch(class(x),
                             "igpuVectorSlice" = "integer",
                             "fgpuVectorSlice" = "float",
                             "dgpuVectorSlice" = "double",
                             stop("gpuVectorSlice type not recognized"))
              return(type)
          })


#' @rdname extract-gpuVector
#' @aliases [,gpuVectorSlice
#' @export
setMethod("[",
          signature(x = "gpuVectorSlice", i = "missing", j = "missing", drop = "missing"),
          function(x, i, j, drop) {
              switch(typeof(x),
                     "integer" = return(get_gpu_slice_vec(x@address, 4L)),
                     "float" = return(get_gpu_slice_vec(x@address, 6L)),
                     "double" = return(get_gpu_slice_vec(x@address, 8L))
              )
          })



