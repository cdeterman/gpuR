#' @import methods

#' @title gpuMatrix Multiplication
#' @param x A gpuMatrix object
#' @param y A gpuMatrix object
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

#' @title gpuMatrix Arith methods
#' @param e1 A gpuMatrix object
#' @param e2 A gpuMatrix object
#' @export
setMethod("Arith", c(e1="gpuMatrix", e2="gpuMatrix"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              switch(op,
                     `+` = gpu_Mat_axpy(1, e1, e2),
                     `-` = gpu_Mat_axpy(-1, e2, e1),
                     {
                         stop("undefined operation")
                     }
              )
          },
valueClass = "gpuMatrix"
)


#' @title Get gpuMatrix type
#' @param x A gpuMatrix object
# @rdname typeof-methods
#' @aliases typeof,gpuMatrix
#' @export
setMethod('typeof', signature(x="gpuMatrix"),
          function(x) {
              switch(class(x),
                     "igpuMatrix" = "integer",
                     "fgpuMatrix" = "float",
                     "dgpuMatrix" = "double")
          })


#' @title gpuMatrix ncol method
#' @param x A gpuMatrix object
#' @export
setMethod('ncol', signature(x="gpuMatrix"),
          function(x) {
              switch(typeof(x),
                     "integer" = return(cpp_incol(x@address)),
                     "float" = return(cpp_fncol(x@address)),
                     "double" = return(cpp_dncol(x@address))
              )
          }
)


#' @title gpuMatrix nrow method
#' @param x A gpuMatrix object
#' @export
setMethod('nrow', signature(x="gpuMatrix"), 
          function(x) {
              switch(typeof(x),
                     "integer" = return(cpp_inrow(x@address)),
                     "float" = return(cpp_fnrow(x@address)),
                     "double" = return(cpp_dnrow(x@address))
              )
          }
)

#' @title gpuMatrix dim method
#' @param x A gpuMatrix object
#' @export
setMethod('dim', signature(x="gpuMatrix"),
          function(x) return(c(nrow(x), ncol(x))))

#' @title Extract all gpuMatrix elements
#' @param x A gpuMatrix object
#' @param i missing
#' @param j missing
#' @param drop missing
#' @export
setMethod("[",
          signature(x = "gpuMatrix", i = "missing", j = "missing", drop = "missing"),
          function(x, i, j, drop) {
              switch(typeof(x),
                     "integer" = return(iXptrToSEXP(x@address)),
                     "float" = return(fXptrToSEXP(x@address)),
                     "double" = return(dXptrToSEXP(x@address))
              )
          })


