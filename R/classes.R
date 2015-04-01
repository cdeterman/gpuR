
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
