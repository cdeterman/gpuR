
#' @rdname extract-methods
#' @export
setMethod("[",
          signature(x = "gpuVector", i = "missing", j = "missing", drop = "missing"),
          function(x, i, j, drop) {
              switch(typeof(x),
                     "integer" = return(getEigenMatrix(x@address, 4L)),
                     "float" = return(getEigenMatrix(x@address, 6L)),
                     "double" = return(getEigenMatrix(x@address, 8L))
              )
          })

#' @rdname extract-methods
#' @export
setMethod("[",
          signature(x = "gpuVector", i = "numeric", j = "missing", drop = "missing"),
          function(x, i, j, drop) {
              
              assert_all_are_in_closed_range(i, lower = 1, upper = length(x))
              
              switch(typeof(x),
                     "integer" = return(GetVecElement(x@address, i, 4L)),
                     "float" = return(GetVecElement(x@address, i, 6L)),
                     "double" = return(GetVecElement(x@address, i, 8L))
              )
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "gpuVector", i = "numeric", j = "missing", value = "numeric"),
          function(x, i, j, value) {
              
              assert_all_are_in_closed_range(i, lower = 1, upper = length(x))
              
              if(length(value) != length(i)){
                  value <- rep(value, length(i))
              }
              
              switch(typeof(x),
                     "float" = SetVecElement(x@address, i, value, 6L),
                     "double" = SetVecElement(x@address, i, value, 8L),
                     stop("type not recongized")
              )
              return(x)
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "gpuVector", i = "numeric", j = "missing", value = "integer"),
          function(x, i, j, value) {
              
              assert_all_are_in_closed_range(i, lower = 1, upper = length(x))
              
              switch(typeof(x),
                     "integer" = SetVecElement(x@address, i, value, 4L),
                     stop("type not recongized")
              )
              return(x)
          })
