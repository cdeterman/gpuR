
#' @rdname extract-methods
#' @export
setMethod("[",
          signature(x = "vclMatrix", i = "missing", j = "missing", drop = "missing"),
          function(x, i, j, drop) {
              
              Rmat <- switch(typeof(x),
                             "integer" = VCLtoMatSEXP(x@address, 4L),
                             "float" = VCLtoMatSEXP(x@address, 6L),
                             "double" = VCLtoMatSEXP(x@address, 8L),
                             "fcomplex" = VCLtoMatSEXP(x@address, 10L),
                             "dcomplex" = VCLtoMatSEXP(x@address, 12L),
                             stop("unsupported matrix type")
              )
              
              return(Rmat)
              
          })

#' @rdname extract-methods
#' @export
setMethod("[",
          signature(x = "vclMatrix", i = "missing", j = "numeric", drop="missing"),
          function(x, i, j, drop) {
              
              type <- switch(typeof(x),
                             "integer" = 4L,
                             "float" = 6L,
                             "double" = 8L,
                             stop("type not recognized")
              )
              
              if(length(j) > 1){
                  
                  out <- matrix(nrow= nrow(x), ncol = length(j))
                  
                  for(c in seq_along(j)){
                      out[,c] <- vclGetCol(x@address, j[c], type, x@.context_index - 1)    
                  }
                  
                  return(out)
                  
              }else{
                  return(vclGetCol(x@address, j, type, x@.context_index - 1))
              }
          })


#' @rdname extract-methods
#' @export
setMethod("[",
          signature(x = "vclMatrix", i = "numeric", j = "missing", drop="missing"),
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
                          out[r,] <- vclGetRow(x@address, i[r], type, x@.context_index - 1)
                      }
                      
                      return(out)
                      
                  }else{
                      return(vclGetRow(x@address, i, type, x@.context_index - 1))    
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
                      
                      output[elem] <- vclGetElement(x@address, row_idx, col_idx, type)
                  }
                  
                  return(output)
              }
              
              # Rmat <- switch(typeof(x),
              #        "integer" = vclGetRow(x@address, i, 4L, x@.context_index - 1),
              #        "float" = vclGetRow(x@address, i, 6L, x@.context_index - 1),
              #        "double" = vclGetRow(x@address, i, 8L, x@.context_index - 1),
              #        stop("unsupported matrix type")
              # )
              
              # return(Rmat)
              
          })

#' @rdname extract-methods
#' @export
setMethod("[",
          signature(x = "vclMatrix", i = "numeric", j = "numeric", drop="missing"),
          function(x, i, j, drop) {
              
              type <- switch(typeof(x),
                             "integer" = 4L,
                             "float" = 6L,
                             "double" = 8L,
                             stop("type not recognized")
              )
              
              if(length(i) > 1 || length(j) > 1){
                  out <- matrix(nrow = length(i), ncol = length(j))
                  
                  for(r in seq_along(i)){
                      for(c in seq_along(j)){
                          out[r,c] <- vclGetElement(x@address, i[r], j[c], type)
                      }
                  }
                  
                  return(out)
              }else{
                  return(vclGetElement(x@address, i, j, type))
              }
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "vclMatrix", i = "missing", j = "numeric", value = "numeric"),
          function(x, i, j, value) {
              
              if(j > ncol(x)){
                  stop("column index exceeds number of columns")
              }
              
              if(length(value) > 1){
                  
                  if(length(value) != nrow(x)){
                      stop("number of items to replace is not a multiple of replacement length")
                  }
                  
                  switch(typeof(x),
                         "float" = vclSetCol(x@address, j, value, 6L),
                         "double" = vclSetCol(x@address, j, value, 8L),
                         stop("unsupported matrix type")
                  )
                  
              }else{
                  switch(typeof(x),
                         "float" = vclFillCol(x@address, j, value, x@.context_index, 6L),
                         "double" = vclFillCol(x@address, j, value, x@.context_index, 8L),
                         stop("unsupported matrix type")
                  )
              }
              
              
              return(x)
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "ivclMatrix", i = "missing", j = "numeric", value = "integer"),
          function(x, i, j, value) {
              
              if(length(value) != nrow(x)){
                  stop("number of items to replace is not a multiple of replacement length")
              }
              
              if(j > ncol(x)){
                  stop("column index exceeds number of columns")
              }
              
              switch(typeof(x),
                     "integer" = vclSetCol(x@address, j, value, 4L),
                     stop("unsupported matrix type")
              )
              
              return(x)
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "vclMatrix", i = "numeric", j = "missing", value = "numeric"),
          function(x, i, j, ..., value) {
              
              assert_all_are_in_closed_range(i, lower = 1, upper = nrow(x))
              
              type <- switch(typeof(x),
                             "integer" = 4L,
                             "float" = 6L,
                             "double" = 8L,
                             stop("type not recognized")
              )
              
              # print(nargs())
              
              if(nargs() == 4){
                  if(length(value) != ncol(x)){
                      stop("number of items to replace is not a multiple of replacement length")
                  }
                  
                  vclSetRow(x@address, i, value, type)
                  
              }else{
                  if(length(value) != length(i)){
                      if(length(value) == 1){
                          value <- rep(value, length(i))
                      }else{
                          stop("number of items to replace is not a multiple of replacement length")
                      }
                  }
                  
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
                      
                      # print(row_idx)
                      # print(col_idx)
                      
                      vclSetElement(x@address, row_idx, col_idx, value[elem], type)
                  }
              }
              
              # 	      if(length(value) != ncol(x)){
              # 	          stop("number of items to replace is not a multiple of replacement length")
              # 	      }
              #               
              #           if(i > nrow(x)){
              #               stop("row index exceeds number of rows")
              #           }
              #           
              #           switch(typeof(x),
              #                  "float" = vclSetRow(x@address, i, value, 6L),
              #                  "double" = vclSetRow(x@address, i, value, 8L),
              #                  stop("unsupported matrix type")
              #           )
              
              return(x)
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "ivclMatrix", i = "numeric", j = "missing", value = "integer"),
          function(x, i, j, value) {
              
              if(length(value) != ncol(x)){
                  stop("number of items to replace is not a multiple of replacement length")
              }
              
              if(i > nrow(x)){
                  stop("row index exceeds number of rows")
              }
              
              switch(typeof(x),
                     "integer" = vclSetRow(x@address, i, value, 4L),
                     stop("unsupported matrix type")
              )
              
              return(x)
          })


#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "vclMatrix", i = "numeric", j = "numeric", value = "numeric"),
          function(x, i, j, value) {
              
              assert_all_are_in_closed_range(i, lower = 1, upper=nrow(x))
              assert_all_are_in_closed_range(j, lower = 1, upper=ncol(x))
              assert_is_scalar(value)
              
              switch(typeof(x),
                     "float" = vclSetElement(x@address, i, j, value, 6L),
                     "double" = vclSetElement(x@address, i, j, value, 8L),
                     stop("unsupported matrix type")
              )
              
              return(x)
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "ivclMatrix", i = "numeric", j = "numeric", value = "integer"),
          function(x, i, j, value) {
              
              assert_all_are_in_closed_range(i, lower = 1, upper=nrow(x))
              assert_all_are_in_closed_range(j, lower = 1, upper=ncol(x))
              assert_is_scalar(value)
              
              switch(typeof(x),
                     "integer" = vclSetElement(x@address, i, j, value, 4L),
                     stop("unsupported matrix type")
              )
              
              return(x)
          })


#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "vclMatrix", i = "missing", j = "missing", value = "matrix"),
          function(x, i, j, value) {
              
              assert_is_matrix(value)
              
              switch(typeof(x),
                     "integer" = vclSetMatrix(x@address, value, 4L, x@.context_index - 1),
                     "float" = vclSetMatrix(x@address, value, 6L, x@.context_index - 1),
                     "double" = vclSetMatrix(x@address, value, 8L, x@.context_index - 1),
                     stop("unsupported matrix type")
              )
              
              return(x)
          })


#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "vclMatrix", i = "missing", j = "missing", value = "vclMatrix"),
          function(x, i, j, value) {
              
              switch(typeof(x),
                     "integer" = vclSetVCLMatrix(x@address, value@address, 4L, x@.context_index - 1),
                     "float" = vclSetVCLMatrix(x@address, value@address, 6L, x@.context_index - 1),
                     "double" = vclSetVCLMatrix(x@address, value@address, 8L, x@.context_index - 1),
                     stop("unsupported matrix type")
              )
              
              return(x)
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "vclMatrix", i = "missing", j = "numeric", value = "vclMatrix"),
          function(x, i, j, value) {
              
              start = head(j, 1) - 1
              end = tail(j, 1)
              
              switch(typeof(x),
                     "integer" = vclMatSetVCLCols(x@address, value@address, start, end, 4L, x@.context_index - 1),
                     "float" = vclMatSetVCLCols(x@address, value@address, start, end, 6L, x@.context_index - 1),
                     "double" = vclMatSetVCLCols(x@address, value@address, start, end, 8L, x@.context_index - 1),
                     stop("unsupported matrix type")
              )
              
              return(x)
          })


#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "vclMatrix", i = "missing", j = "missing", value = "numeric"),
          function(x, i, j, value) {
              
              assert_is_scalar(value)
              
              switch(typeof(x),
                     "integer" = vclFillVCLMatrix(x@address, value, 4L, x@.context_index - 1),
                     "float" = vclFillVCLMatrix(x@address, value, 6L, x@.context_index - 1),
                     "double" = vclFillVCLMatrix(x@address, value, 8L, x@.context_index - 1),
                     stop("unsupported matrix type")
              )
              
              return(x)
          })


#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "vclMatrix", i = "missing", j = "missing", value = "vclVector"),
          function(x, i, j, value) {
              
              switch(typeof(x),
                     "integer" = assignVectorToMat(x@address, value@address, 4L),
                     "float" = assignVectorToMat(x@address, value@address, 6L),
                     "double" = assignVectorToMat(x@address, value@address, 8L),
                     stop("unsupported matrix type")
              )
              
              return(x)
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "vclMatrix", i = "missing", j = "numeric", value = "vclVector"),
          function(x, i, j, value) {
              
              switch(typeof(x),
                     "integer" = assignVectorToCol(x@address, value@address, j-1, 4L),
                     "float" = assignVectorToCol(x@address, value@address, j-1, 6L),
                     "double" = assignVectorToCol(x@address, value@address, j-1, 8L),
                     stop("unsupported matrix type")
              )
              
              return(x)
          })