

#' @rdname as.gpuVector-methods
#' @aliases as.gpuVector,vector
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
setMethod("Compare", c(e1="vector", e2="gpuVector"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              switch(op,
                     `==` = {e1 == e2@object},
                    {
                        stop("undefined operation")
                    }
              )
          },
valueClass = "vector"
)

#' @export
setMethod("Compare", c(e1="gpuVector", e2="vector"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              switch(op,
                     `==` = {e1@object == e2},
                    {
                        stop("undefined operation")
                    }
              )
          },
valueClass = "vector"
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

setOldClass("typeof")
#' @export
setMethod('typeof', signature(x="gpuMatrix"),
          function(x) return(x@type))

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

#' @export
setMethod('dim', signature(x="gpuMatrix"),
          function(x) return(c(nrow(x), ncol(x))))


#' @export
setMethod("[",
          signature(x = "gpuMatrix", i="missing", j="missing", drop = "missing"),
          function(x) {
              switch(typeof(x),
                     "integer" = return(iXptrToSEXP(x@address)),
                     "float" = return(fXptrToSEXP(x@address)),
                     "double" = return(dXptrToSEXP(x@address))
              )
          })


#' @export
print.gpuMatrix <- function(x, ..., n = NULL, width = NULL) {
    cat("Source: gpuR Matrix ", dim_desc(x), "\n", sep = "")
    cat("\n")
    
    if(!is.null(n)){
        assert_is_integer(n)   
    }else{
        n <- ifelse(nrow(x) >= 5, 5L, nrow(x))
    }
    if(!is.null(width)){
        assert_is_integer(width)    
    }else{
        width <- ifelse(ncol(x) >= 5, 5L, ncol(x))
    }
    
    if(width > ncol(x)) stop("width greater than number of columns")
    
    tab <- switch(typeof(x),
                  "integer" = truncIntgpuMat(x@address, n, width),
                  "float" = truncFloatgpuMat(x@address, n, width),
                  "double" = truncDoublegpuMat(x@address, n, width)
                  )
    
    block = structure(
        list(table = tab, extra = ncol(x)-width), 
        class = "trunc_gpuTable")
    
    print(block)    
    
    invisible(x)
}


print.trunc_gpuTable <- function(x, ...) {
    if (!is.null(x$table)) {
        print(x$table)
    }
    
    if (x$extra > 0) {
        nvars <- x$extra
        cat("\n", paste0(nvars, " variables not shown"), "\n", sep = "")
    }

    invisible()
}
