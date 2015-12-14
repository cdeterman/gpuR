
# need code to reshape if dimensions differ from input

#' @title Construct a gpuMatrix
#' @description Construct a gpuMatrix of a class that inherits
#' from \code{gpuMatrix}
#' @param data An object that is or can be converted to a 
#' \code{matrix} object
#' @param nrow An integer specifying the number of rows
#' @param ncol An integer specifying the number of columns
#' @param type A character string specifying the type of gpuMatrix.  Default
#' is NULL where type is inherited from the source data type.
#' @param ... Additional method to pass to gpuMatrix methods
#' @return A gpuMatrix object
#' @docType methods
#' @rdname gpuMatrix-methods
#' @author Charles Determan Jr.
#' @export
setGeneric("gpuMatrix", function(data = NA, nrow=NA, ncol=NA, type=NULL, ...){
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
                                    address=sexpToEigenXptr(data, 
                                                            nrow(data),
                                                            ncol(data), 
                                                            4L))
                            },
                            float = {
                                new("fgpuMatrix", 
                                    address=sexpToEigenXptr(data, 
                                                            nrow(data),
                                                            ncol(data), 
                                                            6L))
                            },
                            double = {
                                new("dgpuMatrix",
                                    address = sexpToEigenXptr(data, 
                                                              nrow(data),
                                                              ncol(data), 
                                                              8L))
                            },
                            stop("this is an unrecognized 
                                 or unimplemented data type")
                            )
              
              return(data)
          },
          valueClass = "gpuMatrix")


#' @rdname gpuMatrix-methods
#' @aliases gpuMatrix,missing
setMethod('gpuMatrix', 
          signature(data = 'missing'),
          function(data, nrow=NA, ncol=NA, type=NULL){
              
              if (is.null(type)) type <- getOption("gpuR.default.type")
              
              assert_is_numeric(nrow)
              assert_is_numeric(ncol)
              
              data = switch(type,
                            integer = {
                                new("igpuMatrix", 
                                    address=emptyEigenXptr(nrow, ncol, 4L))
                            },
                            float = {
                                new("fgpuMatrix", 
                                    address=emptyEigenXptr(nrow, ncol, 6L))
                            },
                            double = {
                                new("dgpuMatrix",
                                    address = emptyEigenXptr(nrow, ncol, 8L))
                            },
                            stop("this is an unrecognized 
                                 or unimplemented data type")
              )
              
              return(data)
          },
          valueClass = "gpuMatrix")



#' @rdname gpuMatrix-methods
#' @aliases gpuMatrix,vector
setMethod('gpuMatrix', 
          signature(data = 'vector'),
          function(data, nrow, ncol, type=NULL){
              
              if (is.null(type)) type <- typeof(data)
              
              if(typeof(data) == "logical" | typeof(data) == "character"){
                  stop(paste0(typeof(data), "type is not supported", sep=" "))
              }
              
              assert_is_numeric(nrow)
              assert_is_numeric(ncol)
              
              data = switch(type,
                            integer = {
                                new("igpuMatrix", 
                                    address=sexpVecToEigenXptr(data, nrow, ncol, 4L))
                            },
                            float = {
                                new("fgpuMatrix", 
                                    address=sexpVecToEigenXptr(data, nrow, ncol, 6L))
                            },
                            double = {
                                new("dgpuMatrix",
                                    address = sexpVecToEigenXptr(data, nrow, ncol, 8L))
                            },
                            stop("this is an unrecognized 
                                 or unimplemented data type")
              )
              
              return(data)
          },
          valueClass = "gpuMatrix")

