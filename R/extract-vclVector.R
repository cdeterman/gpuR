
#' @rdname extract-methods
#' @export
setMethod("[",
          signature(x = "vclVector", i = "missing", j = "missing", drop = "missing"),
          function(x, i, j, drop) {
              switch(typeof(x),
                     "integer" = return(VCLtoVecSEXP(x@address, 4L)),
                     "float" = return(VCLtoVecSEXP(x@address, 6L)),
                     "double" = return(VCLtoVecSEXP(x@address, 8L))
              )
          })


#' @rdname extract-methods
#' @export
setMethod("[",
          signature(x = "vclVector", i = "numeric", j = "missing", drop = "missing"),
          function(x, i, j, drop) {
              
              assert_all_are_in_closed_range(i, lower = 1, upper = length(x))
              
              switch(typeof(x),
                     "integer" = return(vclVecGetElement(x@address, i, 4L)),
                     "float" = return(vclVecGetElement(x@address, i, 6L)),
                     "double" = return(vclVecGetElement(x@address, i, 8L))
              )
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "vclVector", i = "numeric", j = "missing", value="numeric"),
          function(x, i, j, value) {
              
              assert_all_are_positive(i)
              
              if(length(value) > 1 & length(value) != length(i)){
                  stop("number of items to replace is not a multiple of replacement length")
              }
              
              if(length(value) == 1 & length(i) == 1){
                  assert_all_are_in_closed_range(i, lower = 1, upper = length(x))
                  
                  switch(typeof(x),
                         "float" = vclVecSetElement(x@address, i, value, 6L),
                         "double" = vclVecSetElement(x@address, i, value, 8L),
                         stop("type not recognized")
                  )
              }else{
                  start <- head(i, 1) - 1
                  end <- tail(i, 1)
                  
                  switch(typeof(x),
                         "integer" = vclFillVectorRangeScalar(x@address, value, start-1, end, 4L, x@.context_index - 1),
                         "float" = vclFillVectorRangeScalar(x@address, value, start-1, end, 6L, x@.context_index - 1),
                         "double" = vclFillVectorRangeScalar(x@address, value, start-1, end, 8L, x@.context_index - 1),
                         stop("unsupported matrix type")
                  )
              }
              
              
              
              return(x)
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "ivclVector", i = "numeric", j = "missing", value="integer"),
          function(x, i, j, value) {
              if(length(value) != 1){
                  stop("number of items to replace is not a multiple of replacement length")
              }
              
              assert_all_are_in_closed_range(i, lower = 1, upper = length(x))
              
              switch(typeof(x),
                     "integer" = vclVecSetElement(x@address, i, value, 4L),
                     stop("type not recognized")
              )
              return(x)
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "vclVector", i = "logical", j = "missing", value="numeric"),
          function(x, i, j, value) {
              
              f <- NULL
              
              # idx <- 
              # if(length(i) != length(x)){
              #     for(e in seq_along(i)){
              #         seq(from = e, to = length(x), by = length(i)) - 1
              #     }
              # }
              # start <- head(i, 1) - 1
              # end <- tail(i, 1)
              
              if(length(value) == 1) f <- "scalar"
              if(is.null(f)){
                  if(length(value) > 1 && length(value) < length(x)){
                      f <- "slice"
                  }else{
                      if(length(value) != length(i)){
                          stop("number of items to replace is not a multiple of replacement length")
                      }
                      f <- "elementwise"
                  }
              }
              
              switch(f,
                     "scalar" = {
                         switch(typeof(x),
                                "integer" = vclFillVectorScalar(x@address, value, 4L, x@.context_index - 1),
                                "float" = vclFillVectorScalar(x@address, value, 6L, x@.context_index - 1),
                                "double" = vclFillVectorScalar(x@address, value, 8L, x@.context_index - 1),
                                stop("unsupported vector type")
                         )
                     },
                     "slice" = {
                         
                         stop("not fully implemented yet")
                         starts <- which(i)
                         stride <- length(i)
                         
                         switch(typeof(x),
                                "integer" = vclFillVectorSliceScalar(x@address, value, starts - 1L, stride, 4L, x@.context_index - 1L),
                                "float" = vclFillVectorSliceScalar(x@address, value, starts - 1L, stride, 6L, x@.context_index - 1L),
                                "double" = vclFillVectorSliceScalar(x@address, value, starts - 1L, stride, 8L, x@.context_index - 1L),
                                stop("unsupported vector type")
                         )
                     },
                     "elementwise" = {
                         
                         elems <- which(i) - 1L
                         
                         switch(typeof(x),
                                "integer" = vclFillVectorElementwise(x@address, value, elems, 4L, x@.context_index - 1L),
                                "float" = vclFillVectorElementwise(x@address, value, elems, 6L, x@.context_index - 1L),
                                "double" = vclFillVectorElementwise(x@address, value, elems, 8L, x@.context_index - 1L),
                                stop("unsupported vector type")
                         )
                     },
                     stop("Internal Error: no replace logic selected")
              )
              
              return(x)
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "vclVector", i = "missing", j = "missing", value="numeric"),
          function(x, i, j, value) {
              
              if(length(value) > 1){
                  if(length(value) > length(x)){
                      stop("number of items to replace is not a multiple of replacement length")
                  }
                  
                  switch(typeof(x),
                         "integer" = vclSetVector(x@address, value, 4L, x@.context_index - 1),
                         "float" = vclSetVector(x@address, value, 6L, x@.context_index - 1),
                         "double" = vclSetVector(x@address, value, 8L, x@.context_index - 1),
                         stop("unsupported vector type")
                  )
              }else{
                  switch(typeof(x),
                         "integer" = vclFillVectorScalar(x@address, value, 4L, x@.context_index - 1),
                         "float" = vclFillVectorScalar(x@address, value, 6L, x@.context_index - 1),
                         "double" = vclFillVectorScalar(x@address, value, 8L, x@.context_index - 1),
                         stop("unsupported vector type")
                  )
              }
              
              
              return(x)
          })


#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "vclVector", i = "missing", j = "missing", value = "vclVector"),
          function(x, i, j, value) {
              
              switch(typeof(x),
                     "integer" = vclSetVCLVector(x@address, value@address, 4L),
                     "float" = vclSetVCLVector(x@address, value@address, 6L),
                     "double" = vclSetVCLVector(x@address, value@address, 8L),
                     stop("unsupported matrix type")
              )
              
              return(x)
          })


#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "vclVector", i = "numeric", j = "missing", value = "vclVector"),
          function(x, i, j, value) {
              
              
              start <- head(i, 1) - 1
              end <- tail(i, 1)
              
              switch(typeof(x),
                     "integer" = vclSetVCLVectorRange(x@address, value@address, start, end, 4L),
                     "float" = vclSetVCLVectorRange(x@address, value@address, start, end, 6L),
                     "double" = vclSetVCLVectorRange(x@address, value@address, start, end, 8L),
                     stop("unsupported matrix type")
              )
              
              return(x)
          })



#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "vclVector", i = "missing", j = "missing", value = "vclMatrix"),
          function(x, i, j, value) {
              
              if(length(x) != length(value)){
                  stop("lengths must match")
              }
              
              switch(typeof(x),
                     "integer" = vclVecSetVCLMatrix(x@address, value@address, 4L),
                     "float" = vclVecSetVCLMatrix(x@address, value@address, 6L),
                     "double" = vclVecSetVCLMatrix(x@address, value@address, 8L),
                     stop("unsupported matrix type")
              )
              
              return(x)
          })


#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "vclVector", i = "numeric", j = "missing", value = "vclMatrix"),
          function(x, i, j, value) {
              
              if(length(i) != length(value)){
                  stop("lengths must match")
              }
              
              start <- head(i, 1) - 1
              end <- tail(i, 1)
              
              switch(typeof(x),
                     "integer" = vclSetVCLMatrixRange(x@address, value@address, start, end, 4L, x@.context_index - 1L),
                     "float" = vclSetVCLMatrixRange(x@address, value@address, start, end, 6L, x@.context_index - 1L),
                     "double" = vclSetVCLMatrixRange(x@address, value@address, start, end, 8L, x@.context_index - 1L),
                     stop("unsupported matrix type")
              )
              
              return(x)
          })