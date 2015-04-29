#' @title Construct a gpuVector
#' @description Construct a gpuVector of a class that inherits
#' from \code{gpuVector}
#' @param data An object that is or can be converted to a 
#' \code{vector}
#' @param type A character string specifying the type of gpuVector.  Default
#' is NULL where type is inherited from the source data type.
#' @param ... Additional method to pass to gpuVector methods
#' @return A gpuVector object
#' @docType methods
#' @rdname gpuVector-methods
#' @author Charles Determan Jr.
#' @export
setGeneric("gpuVector", function(data, type=NULL, ...){
    standardGeneric("gpuVector")
})

#' @rdname gpuVector-methods
#' @aliases gpuVector,vector
setMethod('gpuVector', 
          signature(data = 'vector'),
          function(data, type=NULL){
              
              if (is.null(type)) type <- typeof(data)
              
              data = switch(type,
                            integer = {
                                new("igpuVector", 
                                    object=data)
                            },
                            float = {
                                stop("float type not implemented")
                                new("fgpuVector", 
                                    object=data)
                            },
                            double = {
                                stop("double type not implemented")
                                new("dgpuVector",
                                    object = data)
                            },
                            stop("this is an unrecognized 
                                 or unimplemented data type")
              )
              
              return(data)
          },
          valueClass = "gpuVector")
              
              
# gpuVector <- function(data = NA, type='integer'){
#     if(is(data, "vector")){
#         data = switch(typeof(data),
#                       integer = {
#                           new("igpuVector", object=data)
#                           },
#                       stop("unrecognized data type")
#                       )
#     }
#     return(data)
# }