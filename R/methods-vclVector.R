
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
              if(length(value) != 1){
                  stop("number of items to replace is not a multiple of replacement length")
              }
              
              assert_all_are_in_closed_range(i, lower = 1, upper = length(x))
              
              switch(typeof(x),
                     "float" = vclVecSetElement(x@address, i, value, 6L),
                     "double" = vclVecSetElement(x@address, i, value, 8L),
                     stop("type not recognized")
              )
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

#' @rdname grapes-o-grapes-methods
#' @export
setMethod("%o%", signature(X="vclVector", Y = "vclVector"),
          function(X,Y)
          {
              if( length(X) != length(Y)){
                  stop("Non-conformant arguments")
              }
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
                         e1 = vclVector(rep(e1, length(e2)), type=typeof(e2))
                         vclVec_axpy(1, e1, e2)
                     },
                     `-` = {
                         e1 = vclVector(rep(e1, length(e2)), type=typeof(e2))
                         vclVec_axpy(-1, e2, e1)
                     },
                     `*` = vclVecScalarMult(e2, e1),
                     `/` = {
                         e1 = vclVector(rep(e1, length(e2)), type=typeof(e2))
                         vclVecElemDiv(e1, e2)
                     },
                     `^` = {
                         e1 <- vclVector(rep(e1, length(e2)), type=typeof(e2))
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
                         e2 = vclVector(rep(e2, length(e1)), type=typeof(e1))
                         vclVec_axpy(1, e1, e2)
                     },
                     `-` = {
                         e2 = vclVector(rep(e2, length(e1)), type=typeof(e1))
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
                     "integer" = return(vcl_igpuVec_size(x@address)),
                     "float" = return(vcl_fgpuVec_size(x@address)),
                     "double" = return(vcl_dgpuVec_size(x@address))
              )
              
          }
)

#' @rdname gpuR-deepcopy
setMethod("deepcopy", signature(object ="vclVector"),
          function(object){
              
              out <- switch(typeof(object),
                            "integer" = new("ivclVector",
                                            address = cpp_deepcopy_vclVector(object@address, 4L)),
                            "float" = new("fvclVector", 
                                          address = cpp_deepcopy_vclVector(object@address, 6L)),
                            "double" = new("dvclVector", 
                                           address = cpp_deepcopy_vclVector(object@address, 8L)),
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
                                new("fvclVectorSlice", address = address)
                            },
                            "double" = {
                                address <- cpp_vclVector_slice(object@address, start, end, 8L)
                                new("dvclVectorSlice", address = address)
                            },
                            stop("type not recognized")
              )
              return(ptr)
          })

