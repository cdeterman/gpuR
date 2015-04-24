
# need code to reshape if dimensions differ from input

#' @export
setGeneric("gpuMatrix", function(data = NA, ncol=NA, nrow=NA, type=NULL, ...){
    standardGeneric("gpuMatrix")
})

setMethod('gpuMatrix', 
          signature(data = 'matrix'),
          function(data, type=NULL){
              
              if (is.null(type)) type <- typeof(data)
              
              data = switch(type,
                            integer = {
                                new("igpuMatrix", 
                                    x=data,
                                    type=type)
                            },
                            float = {
                                new("fgpuMatrix", 
                                    x=data,
                                    type=type)
                            },
                            double = {
                                new("dgpuMatrix",
                                    x = data, 
                                    type=type)
                            },
                            stop("this is an unrecognized 
                                 or unimplemented data type")
                            )
              
              return(data)
          },
          valueClass = "gpuMatrix")

