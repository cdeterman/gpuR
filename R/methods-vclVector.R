
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
          function(object, type=NULL, shared = FALSE){
              
              if(!typeof(object) %in% c('integer', 'float', 'double')){
                  stop("unrecognized data type")
              }
              
              ctx_id <- object@.context_index - 1
              
              tmp = vclMatTovclVec(object@address, shared, ctx_id, 6L)
              
              out <- switch(typeof(object),
                     "integer" = return(new("ivclVector", 
                                            address=vclMatTovclVec(object@address, shared, ctx_id, 4L),
                                            .context_index = object@.context_index,
                                            .platform_index = object@.platform_index,
                                            .platform = object@.platform,
                                            .device_index = object@.device_index,
                                            .device = object@.device)),
                     "float" = {
                         print('creating vector')
                         return(new("fvclVector", 
                                          address=vclMatTovclVec(object@address, shared, ctx_id, 6L),
                                          .context_index = object@.context_index,
                                          .platform_index = object@.platform_index,
                                          .platform = object@.platform,
                                          .device_index = object@.device_index,
                                          .device = object@.device))
                         },
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


#' @rdname grapes-times-grapes-methods
#' @export
setMethod("%*%", signature(x="vclVector", y = "vclVector"),
          function(x,y)
          {
              if( length(x) != length(y)){
                  stop("Non-conformant arguments")
              }
              return(gpuVecInnerProd(x, y))
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
              return(gpuVecOuterProd(X, Y))
          },
          valueClass = "vclMatrix"
)

#' @rdname vclMatrix-crossprod
#' @export
setMethod("tcrossprod",
          signature(x = "vclVector", y = "vclVector"),
          function(x, y){
              return(gpuVecOuterProd(x, y))
          },
          valueClass = "vclMatrix")

#' @rdname vclMatrix-crossprod
#' @export
setMethod("tcrossprod",
          signature(x = "vclVector", y = "missing"),
          function(x, y){
              return(gpuVecOuterProd(x, x))
          },
          valueClass = "vclMatrix")

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
                     `+` = gpuVec_axpy(1, e1, e2),
                     `-` = gpuVec_axpy(-1, e2, e1),
                     `*` = gpuVecElemMult(e1, e2),
                     `/` = gpuVecElemDiv(e1,e2),
                     `^` = gpuVecElemPow(e1, e2),
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
                         gpuVec_axpy(1, e1, e2)
                     },
                     `-` = {
                         e1 = vclVector(rep(e1, length(e2)), type=typeof(e2), ctx_id = e2@.context_index)
                         gpuVec_axpy(-1, e2, e1)
                     },
                     `*` = gpuVecScalarMult(e2, e1),
                     `/` = {
                         e1 = vclVector(rep(e1, length(e2)), type=typeof(e2), ctx_id = e2@.context_index)
                         gpuVecElemDiv(e1, e2)
                     },
                     `^` = {
                         e1 <- vclVector(rep(e1, length(e2)), type=typeof(e2), ctx_id = e2@.context_index)
                         gpuVecElemPow(e1, e2)
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
                         gpuVec_axpy(1, e1, e2)
                     },
                     `-` = {
                         e2 = vclVector(rep(e2, length(e1)), type=typeof(e1), ctx_id = e1@.context_index)
                         gpuVec_axpy(-1, e2, e1)
                     },
                     `*` = gpuVecScalarMult(e1, e2),
                     `/` = gpuVecScalarDiv(e1, e2, 0),
                     `^` = gpuVecScalarPow(e1, e2, 0),
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
                     `-` = gpuVector_unary_axpy(e1),
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
                     `+` = gpuMatVec_axpy(1, e1, e2),
                     `-` = gpuMatVec_axpy(-1, e2, e1),
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
                     `sin` = gpuVecElemSin(x),
                     `asin` = gpuVecElemArcSin(x),
                     `sinh` = gpuVecElemHypSin(x),
                     `cos` = gpuVecElemCos(x),
                     `acos` = gpuVecElemArcCos(x),
                     `cosh` = gpuVecElemHypCos(x),
                     `tan` = gpuVecElemTan(x),
                     `atan` = gpuVecElemArcTan(x),
                     `tanh` = gpuVecElemHypTan(x),
                     `log10` = gpuVecElemLog10(x),
                     `exp` = gpuVecElemExp(x),
                     `abs` = gpuVecElemAbs(x),
                     `sqrt` = gpuVecSqrt(x),
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
                  gpuVecElemLog(x) 
              }else{
                  assert_is_numeric(base)
                  gpuVecElemLogBase(x, base)
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

