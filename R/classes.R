
# The primary class for all gpuVector objects

#' @export
setClass('gpuVector',
         representation("VIRTUAL"),
         validity = function(object) {
             
             if( !length(object) > 0 ){
                 return("gpuVector must be a length greater than 0")
             }
             TRUE
         })

#' @export
setClass("igpuVector",
         slots = c(object = "vector"),
         contains = "gpuVector",
         validity = function(object) {
             if( typeof(object@object) != "integer"){
                 return("igpuVector must be of type 'integer'")
             }
             TRUE
         })


# The primary class for all gpuMatrix objects

#' @export
setClass('gpuMatrix', contains = "big.matrix")

#' @export
setClass("igpuMatrix",
         contains = "gpuMatrix",
         validity = function(object) {
             if( typeof(object) != "integer"){
                 return("igpuMatrix must be of type 'integer'")
             }
             TRUE
         })

#' @export
setClass("fgpuMatrix",
         contains = "gpuMatrix",
         validity = function(object) {
             if( typeof(object) != "float"){
                 return("fgpuMatrix must be of type 'float'")
             }
             TRUE
         })

#' @export
setClass("dgpuMatrix",
         contains = "gpuMatrix",
         validity = function(object) {
             if( typeof(object) != "double"){
                 return("dgpuMatrix must be of type 'double'")
             }
             TRUE
         })

# #' @export
# setClass('gpuMatrix',
#          representation(Dim = "integer", Dimnames = "list", "VIRTUAL"),
#          prototype(Dim = 2L, DImnames = list(NULL, NULL)),
#          validity = function(object) {
#              Dim <- object@Dim
#              if (length(Dim) != 2){
#                  return("Dim slot must be of length 2")
#              }
#              if(any(Dim< 0)){
#                  return("Dim slot must contain non-negative values")
#              }
#              
#              Dimnames <- object@Dimnames
#              if (!is.list(Dimnames) || length(Dimnames) != 2){
#                  return("Dimnames slot must be of length 2")
#              }
#              
#              lenDims <- sapply(Dimnames, length)
#              if (lenDims[1] > 0 && lenDims[1] != Dim[1]){
#                  return("'length(Dimnames[[1]])' must equal Dim[1]")
#              }
#              if (lenDims[2] > 0 && lenDims[2] != Dim[2]){
#                  return("'length(Dimnames[[2]])' must equal Dim[2]")
#              }
#              
#              if( !length(object) > 0 ){
#                  return("gpuMatrix must be a length greater than 0")
#              }
#              TRUE
#          })

# #' @export
# setClass("igpuMatrix",
#          slots = c(object = "matrix"),
#          contains = "gpuMatrix",
#          validity = function(object) {
#              
#              if( typeof(object@object) != "integer"){
#                  return("igpuMatrix must be of type 'integer'")
#              }
#              TRUE
#          })
