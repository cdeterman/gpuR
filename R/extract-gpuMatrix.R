#' @title Extract gpuR object elements
#' @description Operators to extract or replace elements
#' @param x A gpuR object
#' @param i indices specifying rows
#' @param j indices specifying columns
#' @param drop missing
#' @param value data of similar type to be added to gpuMatrix object
#' @param ... Additional arguments
#' @docType methods
#' @rdname extract-methods
#' @author Charles Determan Jr.
#' @export
setMethod("[",
          signature(x = "gpuMatrix", i = "missing", j = "missing", drop = "missing"),
          function(x, i, j, drop) {
              switch(typeof(x),
                     "integer" = return(MatXptrToMatSEXP(x@address, 4L)),
                     "float" = return(MatXptrToMatSEXP(x@address, 6L)),
                     "double" = return(MatXptrToMatSEXP(x@address, 8L)),
                     "fcomplex" = return(MatXptrToMatSEXP(x@address, 10L)),
                     "dcomplex" = return(MatXptrToMatSEXP(x@address, 12L)),
                     stop("type not recognized")
              )
          })

#' @rdname extract-methods
#' @export
setMethod("[",
          signature(x = "gpuMatrix", i = "missing", j = "numeric", drop="missing"),
          function(x, i, j, drop) {
              
              type <- switch(typeof(x),
                             "integer" = 4L,
                             "float" = 6L,
                             "double" = 8L,
                             stop("type not recognized")
              )
              
              if(length(j) > 1){
                  out <- matrix(nrow = nrow(x), ncol = length(j))
                  for(c in seq_along(j)){
                      out[,c] <- GetMatCol(x@address, j[c], type)
                  }
                  return(out)
              }else{
                  return(GetMatCol(x@address, j, type))
              }
          })


#' @rdname extract-methods
#' @export
setMethod("[",
          signature(x = "gpuMatrix", i = "numeric", j = "missing", drop="missing"),
          function(x, i, j, ..., drop) {
              
              if(tail(i, 1) > length(x)){
                  stop("Index out of bounds")
              }
              
              type <- switch(typeof(x),
                             "integer" = 4L,
                             "float" = 6L,
                             "double" = 8L,
                             stop("type not recognized")
              )
              
              if(nargs() == 3){
                  if(length(i) > 1){
                      out <- matrix(nrow = length(i), ncol = ncol(x))
                      
                      for(r in seq_along(i)){
                          out[r,] <- GetMatRow(x@address, i[r], type)
                      }
                      return(out)
                  }else{
                      return(GetMatRow(x@address, i, type))    
                  }
              }else{
                  
                  output <- vector(ifelse(type == 4L, "integer", "numeric"), length(i))
                  
                  nr <- nrow(x)
                  col_idx <- 1
                  for(elem in seq_along(i)){
                      if(i[elem] > nr){
                          tmp <- ceiling(i[elem]/nr)
                          if(tmp != col_idx){
                              col_idx <- tmp
                          }
                          
                          row_idx <- i[elem] - (nr * (col_idx - 1))
                          
                      }else{
                          row_idx <- i[elem]
                      }
                      
                      output[elem] <- GetMatElement(x@address, row_idx, col_idx, type)
                  }
                  
                  return(output)
              }
          })

#' @rdname extract-methods
#' @export
setMethod("[",
          signature(x = "gpuMatrix", i = "numeric", j = "numeric", drop="missing"),
          function(x, i, j, drop) {
              
              type <- switch(typeof(x),
                             "integer" = 4L,
                             "float" = 6L,
                             "double" = 8L,
                             stop("type not recognized")
              )
              
              if(length(i) > 1 || length(j) > 1){
                  
                  out <- matrix(nrow = length(i), ncol=length(j))
                  for(r in seq_along(i)){
                      for(c in seq_along(j)){
                          out[r,c] <- GetMatElement(x@address, i[r], j[c], type)   
                      }
                  }
                  
                  return(out)
                  
              }else{
                  return(GetMatElement(x@address, i, j, type))
              }
              
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "gpuMatrix", i = "numeric", j = "missing", value="numeric"),
          function(x, i, j, ..., value) {
              
              type <- switch(typeof(x),
                             "integer" = 4L,
                             "float" = 6L,
                             "double" = 8L,
                             stop("type not recognized")
              )
              
              if(nargs() == 4){
                  assert_all_are_in_closed_range(i, lower = 1, upper = nrow(x))
                  
                  if(length(value) != ncol(x)){
                      stop("number of items to replace is not a multiple of replacement length")
                  }
                  
                  SetMatRow(x@address, i, value, type)
                  
              }else{
                  if(length(value) != length(i)){
                      if(length(value) == 1){
                          value <- rep(value, length(i))
                      }else{
                          stop("number of items to replace is not a multiple of replacement length")
                      }
                  }
                  
                  
                  output <- vector(ifelse(type == 4L, "integer", "numeric"), length(i))
                  
                  nr <- nrow(x)
                  col_idx <- 1
                  for(elem in seq_along(i)){
                      if(i[elem] > nr){
                          tmp <- ceiling(i[elem]/nr)
                          if(tmp != col_idx){
                              col_idx <- tmp
                          }
                          
                          row_idx <- i[elem] - (nr * (col_idx - 1))
                          
                      }else{
                          row_idx <- i[elem]
                      }
                      
                      SetMatElement(x@address, row_idx, col_idx, value[elem], type)
                  }
              }
              
              return(x)
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "igpuMatrix", i = "numeric", j = "missing", value="integer"),
          function(x, i, j, value) {
              if(length(value) != ncol(x)){
                  stop("number of items to replace is not a multiple of replacement length")
              }
              
              assert_all_are_in_closed_range(i, lower = 1, upper = nrow(x))
              
              switch(typeof(x),
                     "integer" = SetMatRow(x@address, i, value, 4L),
                     stop("type not recognized")
              )
              return(x)
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "gpuMatrix", i = "missing", j = "numeric", value="numeric"),
          function(x, i, j, value) {
              
              if(length(value) != nrow(x)){
                  stop("number of items to replace is not a multiple of replacement length")
              }
              
              assert_all_are_in_closed_range(j, lower = 1, upper = ncol(x))
              
              switch(typeof(x),
                     "float" = SetMatCol(x@address, j, value, 6L),
                     "double" = SetMatCol(x@address, j, value, 8L),
                     stop("type not recognized")
              )
              return(x)
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "igpuMatrix", i = "missing", j = "numeric", value="integer"),
          function(x, i, j, value) {
              
              if(length(value) != nrow(x)){
                  stop("number of items to replace is not a multiple of replacement length")
              }
              
              assert_all_are_in_closed_range(j, lower = 1, upper = ncol(x))
              
              switch(typeof(x),
                     "integer" = SetMatCol(x@address, j, value, 4L),
                     stop("type not recognized")
              )
              return(x)
          })


#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "gpuMatrix", i = "numeric", j = "numeric", value="numeric"),
          function(x, i, j, value) {
              
              assert_all_are_in_closed_range(i, lower = 1, upper = nrow(x))
              assert_all_are_in_closed_range(j, lower = 1, upper = ncol(x))
              
              switch(typeof(x),
                     "float" = SetMatElement(x@address, i, j, value, 6L),
                     "double" = SetMatElement(x@address, i, j, value, 8L),
                     stop("type not recognized")
              )
              return(x)
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "igpuMatrix", i = "numeric", j = "numeric", value="integer"),
          function(x, i, j, value) {
              
              assert_all_are_in_closed_range(i, lower = 1, upper = nrow(x))
              assert_all_are_in_closed_range(j, lower = 1, upper = ncol(x))
              
              switch(typeof(x),
                     "integer" = SetMatElement(x@address, i, j, value, 4L),
                     stop("type not recognized")
              )
              return(x)
          })