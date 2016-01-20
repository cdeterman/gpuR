#' @title Get gpuR object type
#' @description \code{typeof} determines the type (i.e. storage mode) of a 
#' gpuR object
#' @param x A gpuR object
#' @rdname typeof-gpuR-methods
#' @author Charles Determan Jr.
#' @export
setMethod('typeof', signature(x="gpuMatrix"),
          function(x) {
              switch(class(x),
                     "igpuMatrix" = "integer",
                     "fgpuMatrix" = "float",
                     "dgpuMatrix" = "double",
                     "igpuMatrixBlock" = "integer",
                     "fgpuMatrixBlock" = "float",
                     "dgpuMatrixBlock" = "double",
                     stop("unrecognized class"))
          })

#' @rdname typeof-gpuR-methods
#' @export
setMethod('typeof', signature(x="gpuVector"),
          function(x) {
              switch(class(x),
                     "igpuVector" = "integer",
                     "fgpuVector" = "float",
                     "dgpuVector" = "double",
                     "igpuVectorSlice" = "integer",
                     "fgpuVectorSlice" = "float",
                     "dgpuVectorSlice" = "double",
                     stop("unrecognized gpuVector class"))
          })

#' @rdname typeof-gpuR-methods
#' @export
setMethod('typeof', signature(x="vclMatrix"),
          function(x) {
              switch(class(x),
                     "ivclMatrix" = "integer",
                     "fvclMatrix" = "float",
                     "dvclMatrix" = "double",
                     "ivclMatrixBlock" = "integer",
                     "fvclMatrixBlock" = "float",
                     "dvclMatrixBlock" = "double",
                     stop("unrecognized class"))
          })


#' @rdname typeof-gpuR-methods
#' @export
setMethod('typeof', signature(x="vclVector"),
          function(x) {
              switch(class(x),
                     "ivclVector" = "integer",
                     "fvclVector" = "float",
                     "dvclVector" = "double",
                     "ivclVectorSlice" = "integer",
                     "fvclVectorSlice" = "float",
                     "dvclVectorSlice" = "double",
                     stop("unrecognized vclVector class"))
          })


