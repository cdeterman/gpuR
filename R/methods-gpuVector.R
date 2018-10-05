
#' @export
as.vector.gpuVector <- function(x, mode = "any"){
    out <- x[]
    return(out)
}


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

#' @rdname grapes-times-grapes-methods
#' @export
setMethod("%*%", c(x="gpuVector", y="gpuVector"),
          function(x, y){
              if(length(x) != length(y)){
                  stop("non-conformable arguments")
              }
              
              gpuVecInnerProd(x,y)
          })

#' @rdname grapes-times-grapes-methods
#' @export
setMethod("%*%", signature(x="gpuVector", y = "gpuMatrix"),
          function(x,y)
          {
              # print(length(x))
              # print(nrow(y))
              if(length(x) != nrow(y)){
                  stop("Non-conformable arguments")
              }
              return(vclGEMV(x, y))
          },
          valueClass = "gpuVector"
)

#' @title Outer Product
#' @description The outer product of two gpuR vector objects
#' @param X A gpuR object
#' @param Y A gpuR object
#' @docType methods
#' @rdname grapes-o-grapes-methods
#' @author Charles Determan Jr.
#' @export
setMethod("%o%", c(X="gpuVector", Y="gpuVector"),
          function(X, Y){
              gpuVecOuterProd(X,Y)
          })

#' @rdname vclMatrix-crossprod
#' @export
setMethod("tcrossprod",
          signature(x = "gpuVector", y = "gpuVector"),
          function(x, y){
              return(gpuVecOuterProd(x, y))
          },
          valueClass = "gpuMatrix")

#' @rdname vclMatrix-crossprod
#' @export
setMethod("tcrossprod",
          signature(x = "gpuVector", y = "missing"),
          function(x, y){
              return(gpuVecOuterProd(x, x))
          },
          valueClass = "gpuMatrix")

#' @rdname Arith-methods
#' @aliases Arith-gpuVector-gpuVector-method
#' @export
setMethod("Arith", c(e1="gpuVector", e2="gpuVector"),
          function(e1, e2)
          {
              if(length(e1) != length(e2)){
                  stop("non-conformable arguments")
              }
              
              op = .Generic[[1]]
              switch(op,
                     `+` = gpuVec_axpy(1, e1, e2),
                     `-` = gpuVec_axpy(-1, e2, e1),
                     `*` = gpuVecElemMult(e1, e2),
                     `/` = gpuVecElemDiv(e1, e2),
                     `^` = gpuVecElemPow(e1, e2),
                     stop("undefined operation")
                     )
          },
          valueClass = "gpuVector"
)

#' @rdname Arith-methods
#' @aliases Arith-numeric-gpuVector-method
#' @export
setMethod("Arith", c(e1="numeric", e2="gpuVector"),
          function(e1, e2)
          {
              assert_is_of_length(e1, 1)
              
              op = .Generic[[1]]
              switch(op,
                     `+` = {
                         e1 = gpuVector(rep(e1, length(e2)), type=typeof(e2), ctx_id = e2@.context_index)
                         gpuVec_axpy(1, e1, e2)
                         },
                     `-` = {
                         e1 = gpuVector(rep(e1, length(e2)), type=typeof(e2), ctx_id = e2@.context_index)
                         gpuVec_axpy(-1, e2, e1)
                         },
                     `*` = gpuVecScalarMult(e2, e1),
                     `/` = gpuVecScalarDiv(e2, e1, 1),
                     `^` = gpuVecScalarPow(e2, e1, 1),
                     stop("undefined operation")
              )
          },
          valueClass = "gpuVector"
)

#' @rdname Arith-methods
#' @aliases Arith-gpuVector-numeric-method
#' @export
setMethod("Arith", c(e1="gpuVector", e2="numeric"),
          function(e1, e2)
          {
              assert_is_of_length(e2, 1)
              
              op = .Generic[[1]]
              switch(op,
                     `+` = {
                         e2 = gpuVector(rep(e2, length(e1)), type=typeof(e1), ctx_id = e1@.context_index)
                         gpuVec_axpy(1, e1, e2)
                         },
                     `-` = {
                         e2 = gpuVector(rep(e2, length(e1)), type=typeof(e1), ctx_id = e1@.context_index)
                         gpuVec_axpy(-1, e2, e1)
                         },
                     `*` = gpuVecScalarMult(e1, e2),
                     `/` = gpuVecScalarDiv(e1, e2, 0),
                     `^` = gpuVecScalarPow(e1, e2, 0),
                     stop("undefined operation")
              )
          },
          valueClass = "gpuVector"
)

#' @rdname Arith-methods
#' @aliases Arith-gpuVector-missing-method
#' @export
setMethod("Arith", c(e1="gpuVector", e2="missing"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              switch(op,
                     `-` = gpuVector_unary_axpy(e1),
                     stop("undefined operation")
              )
          },
          valueClass = "gpuVector"
)

#' @rdname Arith-methods
#' @export
setMethod("Arith", c(e1="gpuVector", e2="gpuMatrix"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              
              switch(op,
                     `+` = gpuMatVec_axpy(1, e1, e2),
                     `-` = gpuMatVec_axpy(-1, e2, e1),
                     stop("undefined operation")
              )
          },
          valueClass = "gpuMatrix"
)


#' @rdname Math-methods
#' @export
setMethod("Math", c(x="gpuVector"),
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
          valueClass = "gpuVector"
)

#' @rdname log-methods
#' @export
setMethod("log", c(x="gpuVector"),
          function(x, base=NULL)
          {
              if(is.null(base)){
                  gpuVecElemLog(x) 
              }else{
                  assert_is_numeric(base)
                  gpuVecElemLogBase(x, base)
              }
              
          },
          valueClass = "gpuVector"
)

#' @rdname Summary-methods
#' @export
setMethod("Summary", c(x="gpuVector"),
          function(x, ..., na.rm)
          {              
              op = .Generic
              result <- switch(op,
                               `max` = gpuVecMax(x),
                               `min` = gpuVecMin(x),
                               stop("undefined operation")
              )
              return(result)
          }
)

# These compare functions need improvement to have
# a C++ backend function to make faster and more efficient

#' @title Compare vector and gpuVector elements
#' @description Methods for comparison operators
#' @param e1 A vector/gpuVector object
#' @param e2 A vector/gpuVector object
#' @return A logical vector
#' @docType methods
#' @rdname Compare-methods
#' @aliases Compare-vector-gpuVector
#' @author Charles Determan Jr.
#' @export
setMethod("Compare", c(e1="vector", e2="gpuVector"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              switch(op,
                     `==` = {e1 == e2[]},
                     stop("undefined operation")
              )
          },
          valueClass = "vector"
)

#' @rdname Compare-methods
#' @aliases Compare-gpuVector-vector
#' @export
setMethod("Compare", c(e1="gpuVector", e2="vector"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              switch(op,
                     `==` = {e1[] == e2},
{
    stop("undefined operation")
}
              )
          },
valueClass = "vector"
)


# setOldClass("length")

#' @rdname length-methods
#' @export
setMethod('length', signature(x = "gpuVector"),
          function(x) {
              switch(typeof(x),
                     "integer" = return(cpp_gpuVector_size(x@address, 4L)),
                     "float" = return(cpp_gpuVector_size(x@address, 6L)),
                     "double" = return(cpp_gpuVector_size(x@address, 8L))
              )
              
          }
)


#' @rdname gpuR-slice
setMethod("slice",
          signature(object = "gpuVector", start = "integer", end = "integer"),
          function(object, start, end){
              
          assert_all_are_positive(c(start, end))
          assert_all_are_in_range(c(start, end), lower = 1, upper = length(object)+1)
          
          ptr <- switch(typeof(object),
                        "float" = {
                            address <- sliceGPUvec(object@address, start, end, 6L)
                            new("fgpuVectorSlice", 
                                address = address,
                                .context_index = object@.context_index,
                                .platform_index = object@.platform_index,
                                .platform = object@.platform,
                                .device_index = object@.device_index,
                                .device = object@.device)
                        },
                        "double" = {
                            address <- sliceGPUvec(object@address, start, end, 8L)
                            new("dgpuVectorSlice", 
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

#' @rdname gpuR-deepcopy
setMethod("deepcopy", signature(object ="gpuVector"),
          function(object){
              
              out <- switch(typeof(object),
                            "integer" = new("igpuVector",
                                            address = cpp_deepcopy_gpuVector(
                                                object@address, 4L),
                                            .context_index = object@.context_index,
                                            .platform_index = object@.platform_index,
                                            .platform = object@.platform,
                                            .device_index = object@.device_index,
                                            .device = object@.device),
                            "float" = new("fgpuVector", 
                                          address = cpp_deepcopy_gpuVector(
                                              object@address, 6L),
                                          .context_index = object@.context_index,
                                          .platform_index = object@.platform_index,
                                          .platform = object@.platform,
                                          .device_index = object@.device_index,
                                          .device = object@.device),
                            "double" = new("dgpuVector", 
                                           address = cpp_deepcopy_gpuVector(
                                               object@address, 8L),
                                           .context_index = object@.context_index,
                                           .platform_index = object@.platform_index,
                                           .platform = object@.platform,
                                           .device_index = object@.device_index,
                                           .device = object@.device),
                            stop("unrecognized type")
              )
              return(out)
              
          })
