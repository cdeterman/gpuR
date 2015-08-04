
# need code to reshape if dimensions differ from input

#' @title Construct a vclMatrix
#' @description Construct a vclMatrix of a class that inherits
#' from \code{vclMatrix}
#' @param data An object that is or can be converted to a 
#' \code{matrix} object
#' @param nrow An integer specifying the number of rows
#' @param ncol An integer specifying the number of columns
#' @param type A character string specifying the type of vclMatrix.  Default
#' is NULL where type is inherited from the source data type.
#' @param ... Additional method to pass to vclMatrix methods
#' @return A vclMatrix object
#' @docType methods
#' @rdname vclMatrix-methods
#' @author Charles Determan Jr.
#' @export
setGeneric("vclMatrix", function(data = NA, nrow=NA, ncol=NA, type=NULL, ...){
    standardGeneric("vclMatrix")
})

#' @rdname vclMatrix-methods
#' @aliases vclMatrix,matrix
setMethod('vclMatrix', 
          signature(data = 'matrix'),
          function(data, type=NULL){
              
              if (is.null(type)) type <- typeof(data)
              
              data = switch(type,
                            integer = {
                                new("ivclMatrix", 
                                    address=matrixToIntVCL(data))
                            },
                            float = {
                                new("fvclMatrix", 
                                    address=matrixToFloatVCL(data))
                            },
                            double = {
                                new("dvclMatrix",
                                    address = matrixToDoubleVCL(data))
                            },
                            stop("this is an unrecognized 
                                 or unimplemented data type")
                            )
              
              return(data)
          },
          valueClass = "vclMatrix")


#' @rdname vclMatrix-methods
#' @aliases vclMatrix,missing
setMethod('vclMatrix', 
          signature(data = 'missing'),
          function(data, nrow=NA, ncol=NA, type=NULL){
              
              if (is.null(type)) type <- getOption("gpuR.default.type")
              
              data = switch(type,
                            integer = {
                                new("ivclMatrix", 
                                    address=emptyIntVCL(nrow, ncol))
                            },
                            float = {
                                new("fvclMatrix", 
                                    address=emptyFloatVCL(nrow, ncol))
                            },
                            double = {
                                new("dvclMatrix",
                                    address = emptyDoubleVCL(nrow, ncol))
                            },
                            stop("this is an unrecognized 
                                 or unimplemented data type")
                            )
              
              return(data)
          },
          valueClass = "vclMatrix")



# #' @rdname vclMatrix-methods
# #' @aliases vclMatrix,vector
# setMethod('vclMatrix', 
#           signature(data = 'vector'),
#           function(data, nrow, ncol, type=NULL){
#               
#               if (is.null(type)) type <- typeof(data)
#               
#               if(typeof(data) == "logical" | typeof(data) == "character"){
#                   stop(paste0(typeof(data), "type is not supported", sep=" "))
#               }
#               
#               data = switch(type,
#                             integer = {
#                                 new("ivclMatrix", 
#                                     address=vectorToIntMatXptr(data, nrow, ncol))
#                             },
#                             float = {
#                                 new("fvclMatrix", 
#                                     address=vectorToFloatMatXptr(data, nrow, ncol))
#                             },
#                             double = {
#                                 new("dvclMatrix",
#                                     address = vectorToDoubleMatXptr(data, nrow, ncol))
#                             },
#                             stop("this is an unrecognized 
#                                  or unimplemented data type")
#               )
#               
#               return(data)
#           },
#           valueClass = "vclMatrix")
