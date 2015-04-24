


setMethod('as.gpuVector', 
          signature(object = 'vector'),
          function(object, type=NULL){
              if(!typeof(object) %in% c('integer', 'double')){
                  stop("unrecognized data type")
              }
              
              gpuVector(object)
          },
          valueClass = "gpuVector")

#' @export
setMethod("Arith", c(e1="igpuVector", e2="igpuVector"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              switch(op,
                     `+` = gpu_vec_add(e1@object, e2@object),
                     `-` = gpu_vec_subtr(e1@object, e2@object),
                     {
                        stop("undefined operation")
                     }
                     )
          },
          valueClass = "gpuVector"
)

#' @export
setMethod("%*%", signature(x="gpuBigMatrix", y = "gpuBigMatrix"),
          function(x,y)
          {
              if( dim(x)[2] != dim(y)[1]){
                  stop("Non-conformant matrices")
              }
              return(gpu_BigMat_mult(x, y))
          },
          valueClass = "gpuMatrix"
)


#' @export
setMethod("%*%", signature(x="gpuMatrix", y = "gpuMatrix"),
          function(x,y)
          {
              if( dim(x)[2] != dim(y)[1]){
                  stop("Non-conformant matrices")
              }
              return(gpu_Mat_mult(x, y))
          },
          valueClass = "gpuMatrix"
)

#' @export
setMethod('typeof', signature(x="gpuMatrix"),
          function(x) {
              return(x@type)
          }
)

#' @export
setMethod('ncol', signature(x="gpuMatrix"),
          function(x) return(ncol(x@x)))

#' @export
setMethod('nrow', signature(x="gpuMatrix"), 
          function(x) return(nrow(x@x)))

#' @export
setMethod('dim', signature(x="gpuMatrix"),
          function(x) return(c(nrow(x), ncol(x))))

# #' @export
# setMethod("%*%", signature(x="matrix", y = "matrix"),
#           function(x,y)
#           {
#               if( dim(x)[2] != dim(y)[1]){
#                   stop("Non-conformant matrices")
#               }
#               return(gpu_mat_mult(x, y))
#           },
#           valueClass = "gpuMatrix"
# )


# #' @export
# setMethod("dim", signature(x="gpuMatrix"),
#           function(x) return(x@Dim)
# )