
# need code to reshape if dimensions differ from input

#' @title Construct a gpuBigMatrix
#' @description Construct a gpuBigMatrix of a class that inherits
#' from \code{gpuBigMatrix}
#' @param data A matrix object that is or can be converted to a 
#' \code{big.matrix} object
#' @param ncol An integer specifying the number of columns
#' @param nrow An integer specifying the number of rows
#' @param type A character string specifying the type of gpuMatrix.  Default
#' is NULL where type is inherited from the source data type.
#' @param ... Additional method to pass to gpuBigMatrix methods
#' @return A gpuBigMatrix object
#' @docType methods
#' @rdname gpuBigMatrix-methods
#' @author Charles Determan Jr.
#' @export
setGeneric("gpuBigMatrix", function(data = NA, ncol=NA, nrow=NA, type=NULL, ...){
    standardGeneric("gpuBigMatrix")
})

#' @import bigmemory
#' @rdname gpuBigMatrix-methods
#' @aliases gpuBigMatrix,matrix
setMethod('gpuBigMatrix', 
          signature(data = 'matrix'),
          function(data, ncol=NA, nrow=NA, type=NULL){
              
              if(!is.na(ncol) | !is.na(nrow)){
                  dm <- dim(data)
                  
                  if(is.na(nrow)) nr <- nrow(data) else nr <- nrow
                  if(is.na(ncol)) nc <- ncol(data) else nc <- ncol
                  
                  if(dm[1] != nr | dm[2] != nc){
                      data <- matrix(as.numeric(data), nrow=nr, ncol=nc)
                  }else{
                      data <- matrix(as.numeric(data), nrow=nr, ncol=nc, dimnames=dimnames(data))
                  }
              }
             
              if (is.null(type)) type <- typeof(data)
              
              data = switch(type,
                            integer = {
                                new("igpuBigMatrix", 
                                    address=as.big.matrix(data, type=type)@address
                                )
                            },
                            float = {
                                new("fgpuBigMatrix", 
                                    address=as.big.matrix(data, type=type)@address
                                )
                            },
                            double = {
                                new("dgpuBigMatrix",
                                    address = as.big.matrix(data, type=type)@address
                                )
                            },
                            stop("this is an unrecognized 
                                 or unimplemented data type")
              )
              
              return(data)
          },
          valueClass = "gpuBigMatrix")

#' @import bigmemory
#' @rdname gpuBigMatrix-methods
#' @aliases gpuBigMatrix,big.matrix
setMethod('gpuBigMatrix', 
          signature(data = 'big.matrix'),
          function(data, ncol=NA, nrow=NA, type=NULL){
              
              if(!is.na(ncol) | !is.na(nrow)){
                  dm <- dim(data)
                  
                  if(is.na(nrow)) nr <- nrow(data) else nr <- nrow
                  if(is.na(ncol)) nc <- ncol(data) else nc <- ncol
                  
                  if(dm[1] != nr | dm[2] != nc){
                      data <- as.big.matrix(matrix(as.numeric(data), nrow=nr, ncol=nc))
                  }else{
                      data <- as.big.matrix(matrix(as.numeric(data), nrow=nr, ncol=nc, dimnames=dimnames(data)))
                  }
              }
              
              if (is.null(type)) type <- typeof(data)
              
              data = switch(type,
                            integer = {
                                new("igpuBigMatrix", 
                                    address=data@address
                                )
                            },
                            float = {
                                new("fgpuBigMatrix", 
                                    address=data@address
                                )
                            },
                            double = {
                                new("dgpuBigMatrix",
                                    address = data@address
                                )
                            },
                            stop("this is an unrecognized 
                                 or unimplemented data type")
              )
              
              return(data)
          },
          valueClass = "gpuBigMatrix")
