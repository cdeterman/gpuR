
#' @title Extract all vclVector elements
#' @param x A vclVector object
#' @param i missing
#' @param j missing
#' @param drop missing
#' @aliases [,vclVector
#' @author Charles Determan Jr.
#' @rdname extract-vclVector
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


#' @rdname extract-vclVector
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


#' @title vclVector Dot Product
#' @param x A vclVector object
#' @param y A vclVector object
#' @return A vclVector
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

#' @title vclVector Outer Product
#' @param X A vclVector object
#' @param Y A vclVector object
#' @return A vclMatrix object
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

#' @title vclVector Arith methods
#' @param e1 A vclVector object
#' @param e2 A vclVector object
#' @return A vclVector object
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
{
    stop("undefined operation")
}
              )
          },
valueClass = "vclVector"
)


#' @title vclVector Math methods
#' @param x A vclVector object
#' @return A vclVector object
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
                     stop("undefined operation")
              )
          },
          valueClass = "vclVector"
)

#' @title vclVector Logarithms
#' @param x A vclVector object
#' @return A vclVector object
#' @param base A positive number (complex not currently supported by OpenCL):
#' the base with respect to which logarithms are computed.  Defaults to the
#' natural log.
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


#' @title Get vclVector type
#' @param x A vclVector object
# @rdname typeof-methods
#' @aliases typeof,vclVector
#' @export
setMethod('typeof', signature(x="vclVector"),
          function(x) {
              switch(class(x),
                     "ivclVector" = "integer",
                     "fvclVector" = "float",
                     "dvclVector" = "double")
          })


#' @rdname length-methods
#' @aliases length,vclVector
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
