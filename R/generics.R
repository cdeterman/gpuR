#' @title Convert object to a gpuVector
#' @description Construct a gpuVector of a class that inherits
#' from \code{gpuVector}
#' @param object An object that is or can be converted to a 
#' \code{vector} object
#' @param type A character string specifying the type of gpuMatrix.  Default
#' is NULL where type is inherited from the source data type.
#' @return A gpuVector object
#' @docType methods
#' @rdname as.gpuVector-methods
#' @author Charles Determan Jr.
#' @export
setGeneric("as.gpuVector", function(object, type){
    standardGeneric("as.gpuVector")
})

#' @title Convert object to a gpuMatrix
#' @description Construct a gpuMatrix of a class that inherits
#' from \code{gpuMatrix}
#' @param object An object that is or can be converted to a 
#' \code{matrix} object
#' @param type A character string specifying the type of gpuMatrix.  Default
#' is NULL where type is inherited from the source data type.
#' @return A gpuMatrix object
#' @docType methods
#' @rdname as.gpuMatrix-methods
#' @author Charles Determan Jr.
#' @export
setGeneric("as.gpuMatrix", function(object, type){
    standardGeneric("as.gpuMatrix")
})