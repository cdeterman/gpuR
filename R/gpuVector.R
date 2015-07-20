#' @title Construct a gpuVector
#' @description Construct a gpuVector of a class that inherits
#' from \code{gpuVector}
#' @param data An object that is or can be converted to a 
#' \code{vector}
#' @param length A non-negative integer specifying the desired length.
#' @param type A character string specifying the type of gpuVector.  Default
#' is NULL where type is inherited from the source data type.
#' @param ... Additional method to pass to gpuVector methods
#' @return A gpuVector object
#' @docType methods
#' @rdname gpuVector-methods
#' @author Charles Determan Jr.
#' @export
setGeneric("gpuVector", function(data, length, type=NULL, ...){
    standardGeneric("gpuVector")
})

#' @rdname gpuVector-methods
#' @aliases gpuVector,vector
setMethod('gpuVector', 
          signature(data = 'vector', length = 'missing'),
          function(data, length, type=NULL){
              
              if (is.null(type)) type <- typeof(data)
              if (!missing(length)) {
                  warning("length argument not currently used when passing
                          in data")
              }
              
              data = switch(type,
                            integer = {
                                new("igpuVector", 
                                    address=vectorToIntXptr(data))
                            },
                            float = {
                                new("fgpuVector", 
                                    address=vectorToFloatXptr(data))
                            },
                            double = {
                                new("dgpuVector",
                                    address = vectorToDoubleXptr(data))
                            },
                            stop("this is an unrecognized 
                                 or unimplemented data type")
              )
              
              return(data)
          },
          valueClass = "gpuVector")


#' @rdname gpuVector-methods
#' @aliases gpuVector,missingOrNULL
setMethod('gpuVector', 
          signature(data = 'missingOrNULL'),
          function(data, length, type=NULL){
              
              if (is.null(type)) type <- getOption("gpuR.default.type")
              if (length <= 0) stop("length must be a positive integer")
              if (!is.integer(length)) stop("length must be a positive integer")
              
              data = switch(type,
                            integer = {
                                new("igpuVector", 
                                    address=emptyVecIntXptr(length))
                            },
                            float = {
                                new("fgpuVector", 
                                    address=emptyVecFloatXptr(length))
                            },
                            double = {
                                new("dgpuVector",
                                    address = emptyVecDoubleXptr(length))
                            },
                            stop("this is an unrecognized 
                                 or unimplemented data type")
              )
              
              return(data)
          },
          valueClass = "gpuMatrix")
