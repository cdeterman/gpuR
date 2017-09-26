
#' @title Compute the Norm of a Matrix
#' @description Computes a matrix norm of x. The norm can be the one \("O"\) norm, 
#' the infinity \("I"\) norm, the Frobenius \("F"\) norm, the maximum modulus \("M"\) 
#' among elements of a matrix, or the “spectral” or "2"-norm, as determined by 
#' the value of type.
#' @param x A gpuR matrix object
#' @param type character string, specifying the type of matrix norm to be computed.
#' @return The matrix norm, a non-negative number
#' @author Charles Determan Jr.
#' @rdname norm-methods
#' @seealso \link[base]{norm}
#' @export
norm <- function(x, type = "O") UseMethod("norm")

# add base method
#' @export
norm.default <- base::norm

# GPU norm

#' @rdname norm-methods
#' @export
norm.vclMatrix <- 
    function(x, type = "O"){
        
        mtype <- typeof(x)
        
        result <- switch(mtype,
                         integer = {cpp_vclMatrix_norm(x@address, type, 4L)},
                         float = {cpp_vclMatrix_norm(x@address, type, 6L)},
                         double = {cpp_vclMatrix_norm(x@address, type, 8L)},
                         stop("type not recognized")
        )
        
        return(result)
    }


#' @rdname norm-methods
#' @export
norm.gpuMatrix <- 
    function(x, type = "O"){
        
        mtype <- typeof(x)
        
        result <- switch(mtype,
                         integer = {cpp_gpuMatrix_norm(x@address, type, 4L)},
                         float = {cpp_gpuMatrix_norm(x@address, type, 6L)},
                         double = {cpp_gpuMatrix_norm(x@address, type, 8L)},
                         stop("type not recognized")
        )
        
        return(result)
    }


