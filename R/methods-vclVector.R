
#' @export
as.vector.vclVector <- function(x, mode = "any"){
    out <- x[]
    return(out)
}

#' #' @rdname as.vclVector-methods
#' #' @param shared Logical indicating if memory should be shared with \code{x}
#' #' @export
#' as.vclVector <- function (data, shared, ...) {
#'     UseMethod("as.vclVector", data)
#' }

#' @rdname as.vclVector-methods
#' @aliases as.vclVector,vector
setMethod('as.vclVector', 
          signature(object = 'vector'),
          function(object, type=NULL){
              if(!typeof(object) %in% c('integer', 'double')){
                  stop("unrecognized data type")
              }
              
              vclVector(object, type = type)
          },
          valueClass = "vclVector")

#' @rdname as.vclVector-methods
#' @param shared Logical indicating if memory should be shared with \code{x}
#' @aliases as.vclVector,vclMatrix
setMethod('as.vclVector', 
          signature(object = 'vclMatrix'),
          function(object, shared, type=NULL){
              if(!typeof(object) %in% c('integer', 'float', 'double')){
                  stop("unrecognized data type")
              }
              
              ctx_id <- object@.context_index - 1
              
              out <- switch(typeof(object),
                     "integer" = return(new("ivclVector", 
                                            address=vclMatTovclVec(object@address, shared, ctx_id, 4L),
                                            .context_index = object@.context_index,
                                            .platform_index = object@.platform_index,
                                            .platform = object@.platform,
                                            .device_index = object@.device_index,
                                            .device = object@.device)),
                     "float" = return(new("fvclVector", 
                                          address=vclMatTovclVec(object@address, shared, ctx_id, 6L),
                                          .context_index = object@.context_index,
                                          .platform_index = object@.platform_index,
                                          .platform = object@.platform,
                                          .device_index = object@.device_index,
                                          .device = object@.device)),
                     "double" = return(new("dvclVector", 
                                           address=vclMatTovclVec(object@address, shared, ctx_id, 8L),
                                           .context_index = object@.context_index,
                                           .platform_index = object@.platform_index,
                                           .platform = object@.platform,
                                           .device_index = object@.device_index,
                                           .device = object@.device))
              )
              return(out)
          },
          valueClass = "vclVector")

#' #' @rdname as.vclVector-methods
#' #' @param shared Logical indicating if memory should be shared with \code{x}
#' #' @aliases as.gpuVector,matrix
#' #' @export
#' as.vclVector.vclMatrix <- function(data, shared = FALSE, ...){
#'     
#'     ctx_id <- data@.context_index - 1
#'     
#'     switch(typeof(data),
#'            "integer" = return(new("ivclVector", 
#'                                   address=vclMatTovclVec(data@address, shared, ctx_id, 4L),
#'                                   .context_index = data@.context_index,
#'                                   .platform_index = data@.platform_index,
#'                                   .platform = data@.platform,
#'                                   .device_index = data@.device_index,
#'                                   .device = data@.device)),
#'            "float" = return(new("fvclVector", 
#'                                 address=vclMatTovclVec(data@address, shared, ctx_id, 6L),
#'                                 .context_index = data@.context_index,
#'                                 .platform_index = data@.platform_index,
#'                                 .platform = data@.platform,
#'                                 .device_index = data@.device_index,
#'                                 .device = data@.device)),
#'            "double" = return(new("dvclVector", 
#'                                  address=vclMatTovclVec(data@address, shared, ctx_id, 8L),
#'                                  .context_index = data@.context_index,
#'                                  .platform_index = data@.platform_index,
#'                                  .platform = data@.platform,
#'                                  .device_index = data@.device_index,
#'                                  .device = data@.device))
#'     )
#' }


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


#' @rdname grapes-times-grapes-methods
#' @export
setMethod("%*%", signature(x="vclVector", y = "vclVector"),
          function(x,y)
          {
              if( length(x) != length(y)){
                  stop("Non-conformant arguments")
              }
              return(vclVecInner(x, y))
          },
          valueClass = "vclVector"
)

#' @rdname grapes-times-grapes-methods
#' @export
setMethod("%*%", signature(x="vclVector", y = "vclMatrix"),
          function(x,y)
          {
              # print(length(x))
              # print(nrow(y))
              if(length(x) != nrow(y)){
                  stop("Non-conformable arguments")
              }
              return(vclGEMV(x, y))
          },
          valueClass = "vclVector"
)

#' @rdname grapes-o-grapes-methods
#' @export
setMethod("%o%", signature(X="vclVector", Y = "vclVector"),
          function(X,Y)
          {
              return(vclVecOuter(X, Y))
          },
          valueClass = "vclMatrix"
)

#' @rdname Arith-methods
#' @export
setMethod("Arith", c(e1="vclVector", e2="vclVector"),
          function(e1, e2)
          {
              if( length(e1) != length(e2)){
                  stop("Non-conformant arguments")
              }
              
              op = .Generic[[1]]
              switch(op,
                     `+` = vclVec_axpy(1, e1, e2),
                     `-` = vclVec_axpy(-1, e2, e1),
                     `*` = vclVecElemMult(e1, e2),
                     `/` = vclVecElemDiv(e1,e2),
                     `^` = vclVecElemPow(e1, e2),
                     stop("undefined operation")
              )
          },
          valueClass = "vclVector"
)

#' @rdname Arith-methods
#' @export
setMethod("Arith", c(e1="numeric", e2="vclVector"),
          function(e1, e2)
          {
              assert_is_of_length(e1, 1)
              
              op = .Generic[[1]]
              switch(op,
                     `+` = {
                         e1 = vclVector(rep(e1, length(e2)), type=typeof(e2), ctx_id = e2@.context_index)
                         vclVec_axpy(1, e1, e2)
                     },
                     `-` = {
                         e1 = vclVector(rep(e1, length(e2)), type=typeof(e2), ctx_id = e2@.context_index)
                         vclVec_axpy(-1, e2, e1)
                     },
                     `*` = vclVecScalarMult(e2, e1),
                     `/` = {
                         e1 = vclVector(rep(e1, length(e2)), type=typeof(e2), ctx_id = e2@.context_index)
                         vclVecElemDiv(e1, e2)
                     },
                     `^` = {
                         e1 <- vclVector(rep(e1, length(e2)), type=typeof(e2), ctx_id = e2@.context_index)
                         vclVecElemPow(e1, e2)
                     },
                     stop("undefined operation")
              )
          },
          valueClass = "vclVector"
)

#' @rdname Arith-methods
#' @export
setMethod("Arith", c(e1="vclVector", e2="numeric"),
          function(e1, e2)
          {
              assert_is_of_length(e2, 1)
              
              op = .Generic[[1]]
              switch(op,
                     `+` = {
                         e2 = vclVector(rep(e2, length(e1)), type=typeof(e1), ctx_id = e1@.context_index)
                         vclVec_axpy(1, e1, e2)
                     },
                     `-` = {
                         e2 = vclVector(rep(e2, length(e1)), type=typeof(e1), ctx_id = e1@.context_index)
                         vclVec_axpy(-1, e2, e1)
                     },
                     `*` = vclVecScalarMult(e1, e2),
                     `/` = vclVecScalarDiv(e1, e2),
                     `^` = vclVecScalarPow(e1, e2),
                     stop("undefined operation")
              )
          },
          valueClass = "vclVector"
)

#' @rdname Arith-methods
#' @export
setMethod("Arith", c(e1="vclVector", e2="missing"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              switch(op,
                     `-` = vclVector_unary_axpy(e1),
                     stop("undefined operation")
              )
          },
          valueClass = "vclVector"
)

#' @rdname Arith-methods
#' @export
setMethod("Arith", c(e1="vclVector", e2="vclMatrix"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              
              switch(op,
                     `+` = vclMatVec_axpy(1, e1, e2),
                     `-` = vclMatVec_axpy(-1, e2, e1),
                     stop("undefined operation")
              )
          },
          valueClass = "vclMatrix"
)

#' @rdname Math-methods
#' @export
setMethod("Math", c(x="vclVector"),
          function(x)
          {
              op = .Generic[[1]]
              switch(op,
                     `sin` = vclVecElemSin(x),
                     `asin` = vclVecElemArcSin(x),
                     `sinh` = vclVecElemHypSin(x),
                     `cos` = vclVecElemCos(x),
                     `acos` = vclVecElemArcCos(x),
                     `cosh` = vclVecElemHypCos(x),
                     `tan` = vclVecElemTan(x),
                     `atan` = vclVecElemArcTan(x),
                     `tanh` = vclVecElemHypTan(x),
                     `log10` = vclVecElemLog10(x),
                     `exp` = vclVecElemExp(x),
                     `abs` = vclVecElemAbs(x),
                     `sqrt` = vclVecSqrt(x),
                     `sign` = gpuVecSign(x),
                     stop("undefined operation")
              )
          },
          valueClass = "vclVector"
)

#' @rdname log-methods
#' @export
setMethod("log", c(x="vclVector"),
          function(x, base=NULL)
          {
              if(is.null(base)){
                  vclVecElemLog(x) 
              }else{
                  assert_is_numeric(base)
                  vclVecElemLogBase(x, base)
              }
              
          },
          valueClass = "vclVector"
)

#' @rdname Summary-methods
#' @export
setMethod("Summary", c(x="vclVector"),
          function(x, ..., na.rm)
          {              
              op = .Generic
              result <- switch(op,
                               `max` = vclVecMax(x),
                               `min` = vclVecMin(x),
                               stop("undefined operation")
              )
              return(result)
          }
)


#' @rdname length-methods
#' @export
setMethod('length', signature(x = "vclVector"),
          function(x) {
              switch(typeof(x),
                     "integer" = return(cpp_vclVector_size(x@address, 4L)),
                     "float" = return(cpp_vclVector_size(x@address, 6L)),
                     "double" = return(cpp_vclVector_size(x@address, 8L))
              )
              
          }
)

#' @rdname gpuR-deepcopy
setMethod("deepcopy", signature(object ="vclVector"),
          function(object){
              
              out <- 
                  switch(typeof(object),
                         "integer" = new("ivclVector",
                                         address = cpp_deepcopy_vclVector(
                                             object@address, 
                                             4L, 
                                             object@.context_index - 1),
                                         .context_index = object@.context_index,
                                         .platform_index = object@.platform_index,
                                         .platform = object@.platform,
                                         .device_index = object@.device_index,
                                         .device = object@.device),
                         "float" = new("fvclVector", 
                                       address = cpp_deepcopy_vclVector(
                                           object@address, 6L, 
                                           object@.context_index - 1),
                                       .context_index = object@.context_index,
                                       .platform_index = object@.platform_index,
                                       .platform = object@.platform,
                                       .device_index = object@.device_index,
                                       .device = object@.device),
                         "double" = new("dvclVector", 
                                        address = cpp_deepcopy_vclVector(
                                            object@address, 8L, 
                                            object@.context_index - 1),
                                        .context_index = object@.context_index,
                                        .platform_index = object@.platform_index,
                                        .platform = object@.platform,
                                        .device_index = object@.device_index,
                                        .device = object@.device),
                         stop("unrecognized type")
                  )
              return(out)
          })

#' @rdname gpuR-slice
setMethod("slice",
          signature(object = "vclVector", start = "integer", end = "integer"),
          function(object, start, end){
              
              assert_all_are_positive(c(start, end))
              assert_all_are_in_range(c(start, end), lower = 1, upper = length(object)+1)
              
              ptr <- switch(typeof(object),
                            "float" = {
                                address <- cpp_vclVector_slice(object@address, start, end, 6L)
                                new("fvclVectorSlice", 
                                    address = address,
                                    .context_index = object@.context_index,
                                    .platform_index = object@.platform_index,
                                    .platform = object@.platform,
                                    .device_index = object@.device_index,
                                    .device = object@.device)
                            },
                            "double" = {
                                address <- cpp_vclVector_slice(object@address, start, end, 8L)
                                new("dvclVectorSlice", 
                                    address = address,
                                    .context_index = object@.context_index,
                                    .platform_index = object@.platform_index,
                                    .platform = object@.platform,
                                    .device_index = object@.device_index,
                                    .device = object@.device)
                            },
                            stop("type not recognized")
              )
              return(ptr)
          })

