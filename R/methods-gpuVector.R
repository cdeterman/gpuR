
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

#' @title gpuVector Arith methods
#' @param e1 A igpuVector object
#' @param e2 A igpuVector object
#' @export
setMethod("Arith", c(e1="igpuVector", e2="igpuVector"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              switch(op,
                     `+` = gpu_vec_add(e1, e2),
                     `-` = gpu_vec_subtr(e1, e2),
{
    stop("undefined operation")
}
              )
          },
valueClass = "gpuVector"
)

#' @title Compare vector and gpuVector elements
#' @param e1 A vector object
#' @param e2 A gpuVector object
#' @export
setMethod("Compare", c(e1="vector", e2="gpuVector"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              switch(op,
                     `==` = {e1 == e2@object},
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
                     `==` = {e1@object == e2},
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
#' @param x A gpuVector object
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

#' @title Extract all gpuVector elements
#' @param x A gpuVector object
#' @param i missing
#' @param j missing
#' @param drop missing
#' @export
setMethod("[",
          signature(x = "gpuVector", i = "missing", j = "missing", drop = "missing"),
          function(x, i, j, drop) {
              switch(typeof(x),
                     "integer" = return(iXptrToVecSEXP(x@address)),
                     "float" = return(fXptrToVecSEXP(x@address)),
                     "double" = return(dXptrToVecSEXP(x@address))
              )
          })
