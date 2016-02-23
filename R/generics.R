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


#' @title Vector Slices
#' @description This doesn't create a copy, it provides a child class that
#' points to a contiguous subvector of a \code{\link{gpuVector}} or
#' \code{\link{vclVector}}.  Non-contiguous slices are currently not supported.
#' @param object A \code{gpuVector} or \code{vclVector} object
#' @param start An integer indicating the start of slice
#' @param end An integer indicating the end of slice
#' @details This function allows a user to create a gpuR vector object that
#' references a continous subset of columns and rows of another gpuR vector
#' object without a copy.  
#' 
#' NOTE - this means that altering values in a vector slice object will alter
#' values in the source vector.
#' @return A \code{gpuVectorSlice} or \code{vclVectorSlice} object
#' @author Charles Determan Jr.
#' @docType methods
#' @name slice
#' @rdname gpuR-slice
#' @aliases slice
#' @export
setGeneric("slice", function(object, start, end){
    standardGeneric("slice")
})

#' @title Matrix Blocks
#' @description This doesn't create a copy, it provides a child class that
#' points to a contiguous submatrix of a \code{\link{gpuMatrix}} or
#' \code{\link{vclMatrix}}.  Non-contiguous blocks are currently not supported.
#' @param object A \code{gpuMatrix} or \code{vclMatrix} object
#' @param rowStart An integer indicating the first row of block
#' @param rowEnd An integer indicating the last row of block
#' @param colStart An integer indicating the first column of block
#' @param colEnd An integer indicating the last column of block
#' @details This function allows a user to create a gpuR matrix object that
#' references a continous subset of columns and rows of another gpuR matrix
#' object without a copy.  
#' 
#' NOTE - this means that altering values in a matrix block object will alter
#' values in the source matrix.
#' @return A \code{gpuMatrixBlock} or \code{vclMatrixBlock} object
#' @author Charles Determan Jr.
#' @docType methods
#' @name block
#' @rdname gpuR-block
#' @aliases block
#' @export
setGeneric("block", function(object, rowStart, rowEnd, colStart, colEnd){
    standardGeneric("block")
})

#' @title Copy a "gpuR" object
#' @description This is needed to make a duplicate of a gpuR object 
#' @param object A gpuR object
#' @details This is needed to make a duplicate of a gpuR object 
#' (i.e. \code{\link{gpuMatrix}}, \code{\link{gpuVector}}, 
#' \code{\link{vclMatrix}}, \code{\link{vclVector}} because
#' the traditional syntax would only copy the pointer of the object.
#' @return A gpuR object
#' @seealso \code{\link{block}}
#' @author Charles Determan Jr.
#' @docType methods
#' @rdname gpuR-deepcopy
#' @export
setGeneric("deepcopy", function(object){
    standardGeneric("deepcopy")
})

# #' @export
# setGeneric("colSums", function(x, na.rm, dims){
#     standardGeneric("colSums")
# }, useAsDefault=base::colSums)

#' @rdname dist-vclMatrix
#' @aliases distance
#' @export
setGeneric("distance", function(x, y, method = "euclidean"){
    standardGeneric("distance")
})
