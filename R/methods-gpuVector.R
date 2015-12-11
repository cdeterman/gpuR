
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

#' @title gpuVector Inner Product
#' @param x A gpuVector object
#' @param y A gpuVector object
#' @return A numeric value
#' @rdname gpuVector-prods
#' @export
setMethod("%*%", c(x="gpuVector", y="gpuVector"),
          function(x, y){
              if(length(x) != length(y)){
                  stop("non-conformable arguments")
              }
              
              gpuVecInnerProd(x,y)
          })

#' @title gpuVector Outer Product
#' @param X A gpuVector object
#' @param Y A gpuVector object
#' @return A gpuMatrix object
#' @export
setMethod("%o%", c(X="gpuVector", Y="gpuVector"),
          function(X, Y){
              if(length(X) != length(Y)){
                  stop("non-conformable arguments")
              }
              
              gpuVecOuterProd(X,Y)
          })

#' @title gpuVector Arith methods
#' @param e1 A gpuVector object
#' @param e2 A gpuVector object
#' @return A gpuVector object
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

#' @title gpuVector Arith methods
#' @param e1 A numeric object
#' @param e2 A gpuVector object
#' @return A gpuVector object
#' @export
setMethod("Arith", c(e1="numeric", e2="gpuVector"),
          function(e1, e2)
          {
              assert_is_of_length(e1, 1)
              
              op = .Generic[[1]]
              switch(op,
                     `+` = {
                         e1 = gpuVector(rep(e1, length(e2)), type=typeof(e2))
                         gpuVec_axpy(1, e1, e2)
                         },
                     `-` = {
                         e1 = gpuVector(rep(e1, length(e2)), type=typeof(e2))
                         gpuVec_axpy(-1, e2, e1)
                         },
                     `*` = gpuVecScalarMult(e2, e1),
                     `/` = {
                         e1 = gpuVector(rep(e1, length(e2)), type=typeof(e2))
                         gpuVecElemDiv(e1, e2)
                         },
                     `^` = {
                         gpuVecScalarPow(e2, e1, 1)
                         #e1 = gpuVector(rep(e1, length(e2)), type=typeof(e2))
                         #gpuVecElemPow(e1, e2)
                         },
                     stop("undefined operation")
              )
          },
          valueClass = "gpuVector"
)

#' @title gpuVector Arith methods
#' @param e1 A gpuVector object
#' @param e2 A numeric object
#' @return A gpuVector object
#' @export
setMethod("Arith", c(e1="gpuVector", e2="numeric"),
          function(e1, e2)
          {
              assert_is_of_length(e2, 1)
              
              op = .Generic[[1]]
              switch(op,
                     `+` = {
                         e2 = gpuVector(rep(e2, length(e1)), type=typeof(e1))
                         gpuVec_axpy(1, e1, e2)
                         },
                     `-` = {
                         e2 = gpuVector(rep(e2, length(e1)), type=typeof(e1))
                         gpuVec_axpy(-1, e2, e1)
                         },
                     `*` = gpuVecScalarMult(e1, e2),
                     `/` = gpuVecScalarDiv(e1, e2),
                     `^` = {
                         gpuVecScalarPow(e1, e2, 0)
                         #e2 = gpuVector(rep(e2, length(e1)), type=typeof(e1))
                         #gpuVecElemPow(e1, e2)
                         },
                     stop("undefined operation")
              )
          },
          valueClass = "gpuVector"
)

#' @title gpuVector Arith methods
#' @param e1 A gpuVector object
#' @param e2 missing
#' @return A gpuVector object
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

#' @title gpuVector Arith methods
#' @param e1 A gpuVectorSlice object
#' @param e2 A gpuVectorSlice object
#' @return A gpuVector object
#' @export
setMethod("Arith", c(e1="gpuVectorSlice", e2="gpuVectorSlice"),
          function(e1, e2)
          {
              if(length(e1) != length(e2)){
                  stop("non-conformable arguments")
              }
              
              op = .Generic[[1]]
              switch(op,
                     `+` = gpuSliceVec_axpy(1, e1, e2),
                     `-` = gpuSliceVec_axpy(-1, e2, e1),
                     stop("undefined operation")
              )
          },
          valueClass = "gpuVector"
)

#' @title gpuVector Math methods
#' @param x A gpuVector object
#' @return A gpuVector object
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
                     stop("undefined operation")
              )
          },
          valueClass = "gpuVector"
)

# These compare functions need improvement to have
# a C++ backend function to make faster and more efficient

#' @title Compare vector and gpuVector elements
#' @param e1 A vector object
#' @param e2 A gpuVector object
#' @export
setMethod("Compare", c(e1="vector", e2="gpuVector"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              switch(op,
                     `==` = {e1 == e2[]},
{
    stop("undefined operation")
}
              )
          },
valueClass = "vector"
)

#' @title Compare gpuVector and vector elements
#' @param e1 A gpuvector object
#' @param e2 A Vector object
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

#' @title Get gpuVector type
#' @param x A gpuVector object
# @rdname typeof-methods
#' @aliases typeof,gpuVector
#' @export
setMethod('typeof', signature(x="gpuVector"),
          function(x) {
              switch(class(x),
                     "igpuVector" = "integer",
                     "fgpuVector" = "float",
                     "dgpuVector" = "double")
          })


# setOldClass("length")

#' @title Length of gpuVector
#' @param x A gpuVector or vclVector object
#' @return A numeric value
#' @rdname length-methods
#' @aliases length,gpuVector
#' @export
setMethod('length', signature(x = "gpuVector"),
          function(x) {
              switch(typeof(x),
                     "integer" = return(cpp_igpuVec_size(x@address)),
                     "float" = return(cpp_fgpuVec_size(x@address)),
                     "double" = return(cpp_dgpuVec_size(x@address))
              )
              
          }
)


#' @title Extract/Set gpuVector elements
#' @param x A gpuVector object
#' @param i missing
#' @param j missing
#' @param drop missing
#' @param value data of similar type to be added to gpuMatrix object
#' @author Charles Determan
#' @rdname extract-gpuVector
#' @aliases [,gpuVector
#' @export
setMethod("[",
          signature(x = "gpuVector", i = "missing", j = "missing", drop = "missing"),
          function(x, i, j, drop) {
              switch(typeof(x),
                     "integer" = return(VecXptrToVecSEXP(x@address, 4L)),
                     "float" = return(VecXptrToVecSEXP(x@address, 6L)),
                     "double" = return(VecXptrToVecSEXP(x@address, 8L))
              )
          })

#' @rdname extract-gpuVector
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

#' @rdname extract-gpuVector
#' @export
setMethod("[<-",
          signature(x = "gpuVector", i = "numeric", j = "missing", value = "numeric"),
          function(x, i, j, value) {
              
              assert_all_are_in_closed_range(i, lower = 1, upper = length(x))
              
              switch(typeof(x),
                     "float" = SetVecElement(x@address, i, value, 6L),
                     "double" = SetVecElement(x@address, i, value, 8L),
                     stop("type not recongized")
              )
              return(x)
          })

#' @rdname extract-gpuVector
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


#' @export
setMethod("slice",
          signature(object = "gpuVector", start = "integer", end = "integer"),
          function(object, start, end){
              
          assert_all_are_positive(c(start, end))
          assert_all_are_in_range(c(start, end), lower = 1, upper = length(object)+1)
          
          ptr <- switch(typeof(object),
                        "float" = {
                            address <- sliceGPUvec(object@address, start, end, 6L)
                            new("fgpuVectorSlice", address = address)
                        },
                        "double" = {
                            address <- sliceGPUvec(object@address, start, end, 8L)
                            new("dgpuVectorSlice", address = address)
                        },
                        stop("type not recognized")
          )
          
          return(ptr)
          
        })

setMethod("deepcopy", signature(object ="gpuVector"),
          function(object){
              
              out <- switch(typeof(object),
                            "integer" = new("igpuVector",
                                            address = cpp_deepcopy_gpuVector(object@address, 4L)),
                            "float" = new("fgpuVector", 
                                          address = cpp_deepcopy_gpuVector(object@address, 6L)),
                            "double" = new("dgpuVector", 
                                           address = cpp_deepcopy_gpuVector(object@address, 8L)),
                            stop("unrecognized type")
              )
              return(out)
              
          })
