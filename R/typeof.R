#' @title Get gpuMatrix type
#' @param x A gpuMatrix object
#' @aliases typeof,gpuMatrix
#' @export
setMethod('typeof', signature(x="gpuMatrix"),
          function(x) {
              switch(class(x),
                     "igpuMatrix" = "integer",
                     "fgpuMatrix" = "float",
                     "dgpuMatrix" = "double")
          })

#' @title Get vclMatrix type
#' @param x A vclMatrix object
#' @aliases typeof,vclMatrix
#' @export
setMethod('typeof', signature(x="vclMatrix"),
          function(x) {
              switch(class(x),
                     "ivclMatrix" = "integer",
                     "fvclMatrix" = "float",
                     "dvclMatrix" = "double")
          })
