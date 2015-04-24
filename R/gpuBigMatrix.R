
# need code to reshape if dimensions differ from input

#' @export
setGeneric("gpuBigMatrix", function(data = NA, ncol=NA, nrow=NA, type=NULL, ...){
    standardGeneric("gpuBigMatrix")
})

#' @import bigmemory
setMethod('gpuBigMatrix', 
          signature(data = 'matrix'),
          function(data, ncol=NA, nrow=NA, type=NULL){
              
              if(!is.na(ncol) | !is.na(nrow)){
                  dm <- dim(data)
                  
                  if(is.na(nrow)) nr <- nrow(data) else nr <- nrow
                  if(is.na(ncol)) nc <- ncol(data) else nc <- ncol
                  
                  if(dim[1] != nr | dim[2] != nc){
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
