#' @title Construct a vclVector
#' @description Construct a vclVector of a class that inherits
#' from \code{vclVector}.  This class points to memory directly on
#' the GPU to avoid the cost of data transfer between host and device.
#' @param data An object that is or can be converted to a 
#' \code{vector}
#' @param length A non-negative integer specifying the desired length.
#' @param type A character string specifying the type of vclVector.  Default
#' is NULL where type is inherited from the source data type.
#' @param ... Additional method to pass to vclVector methods
#' @return A vclVector object
#' @docType methods
#' @rdname vclVector-methods
#' @author Charles Determan Jr.
#' @export
setGeneric("vclVector", function(data, length, type=NULL, ...){
    standardGeneric("vclVector")
})

#' @rdname vclVector-methods
#' @aliases vclVector,vector
setMethod('vclVector', 
          signature(data = 'vector', length = 'missing'),
          function(data, length, type=NULL){
              
              if (is.null(type)) type <- typeof(data)
              if (!missing(length)) {
                  warning("length argument not currently used when passing
                          in data")
              }
              
              data = switch(type,
                            integer = {
                                new("ivclVector", 
                                    address=vectorToIntVCL(data))
                            },
                            float = {
                                new("fvclVector", 
                                    address=vectorToFloatVCL(data))
                            },
                            double = {
                                new("dvclVector",
                                    address = vectorToDoubleVCL(data))
                            },
                            stop("this is an unrecognized 
                                 or unimplemented data type")
                            )
              
              return(data)
              },
          valueClass = "vclVector")


#' @rdname vclVector-methods
#' @aliases vclVector,missing
setMethod('vclVector', 
          signature(data = 'missing'),
          function(data, length, type=NULL){
              
              if (is.null(type)) type <- getOption("gpuR.default.type")
              if (length <= 0) stop("length must be a positive integer")
              if (!is.integer(length)) stop("length must be a positive integer")
              
              data = switch(type,
                            integer = {
                                new("ivclVector", 
                                    address=emptyVecIntVCL(length))
                            },
                            float = {
                                new("fvclVector", 
                                    address=emptyVecFloatVCL(length))
                            },
                            double = {
                                new("dvclVector",
                                    address = emptyVecDoubleVCL(length))
                            },
                            stop("this is an unrecognized 
                                 or unimplemented data type")
              )
              
              return(data)
          },
          valueClass = "vclVector")
