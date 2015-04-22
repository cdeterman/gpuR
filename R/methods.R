


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
setMethod("%*%", signature(x="gpuMatrix", y = "gpuMatrix"),
          function(x,y)
          {
              if( dim(x)[2] != dim(y)[1]){
                  stop("Non-conformant matrices")
              }
              return(gpu_mat_mult(x, y))
          },
          valueClass = "gpuMatrix"
)


# #' @export
# setMethod("dim", signature(x="gpuMatrix"),
#           function(x) return(x@Dim)
# )