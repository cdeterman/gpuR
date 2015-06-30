
# need code to reshape if dimensions differ from input

#' @title Construct a gpuMatrix
#' @description Construct a gpuMatrix of a class that inherits
#' from \code{gpuMatrix}
#' @param data An object that is or can be converted to a 
#' \code{matrix} object
#' @param ncol An integer specifying the number of columns
#' @param nrow An integer specifying the number of rows
#' @param type A character string specifying the type of gpuMatrix.  Default
#' is NULL where type is inherited from the source data type.
#' @param ... Additional method to pass to gpuMatrix methods
#' @return A gpuMatrix object
#' @docType methods
#' @rdname gpuMatrix-methods
#' @author Charles Determan Jr.
#' @export
setGeneric("gpuMatrix", function(data = NA, ncol=NA, nrow=NA, type=NULL, ...){
    standardGeneric("gpuMatrix")
})

#' @rdname gpuMatrix-methods
#' @aliases gpuMatrix,matrix
setMethod('gpuMatrix', 
          signature(data = 'matrix'),
          function(data, type=NULL){
              
              if (is.null(type)) type <- typeof(data)
              
              data = switch(type,
                            integer = {
                                new("igpuMatrix", 
                                    address=matrixToIntXptr(data),
                                    type=type)
                            },
                            float = {
                                new("fgpuMatrix", 
                                    address=matrixToFloatXptr(data),
                                    type=type)
                            },
                            double = {
                                new("dgpuMatrix",
                                    address = matrixToDoubleXptr(data), 
                                    type=type)
                            },
                            stop("this is an unrecognized 
                                 or unimplemented data type")
                            )
              
              return(data)
          },
          valueClass = "gpuMatrix")


#' @rdname gpuMatrix-methods
#' @aliases gpuMatrix,missingOrNULL
setMethod('gpuMatrix', 
          signature(data = 'missingOrNULL'),
          function(data, ncol=NA, nrow=NA, type=NULL){
              
              if (is.null(type)) type <- getOption("gpuR.default.type")
              
              data = switch(type,
                            integer = {
                                new("igpuMatrix", 
                                    address=emptyIntXptr(nrow, ncol),
                                    type=type)
                            },
                            float = {
                                new("fgpuMatrix", 
                                    address=emptyFloatXptr(nrow, ncol),
                                    type=type)
                            },
                            double = {
                                new("dgpuMatrix",
                                    address = emptyDoubleXptr(nrow, ncol), 
                                    type=type)
                            },
                            stop("this is an unrecognized 
                                 or unimplemented data type")
              )
              
              return(data)
          },
          valueClass = "gpuMatrix")
