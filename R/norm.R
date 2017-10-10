
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
setMethod("norm", signature(x = "vclMatrix", type = "character"),
    function(x, type){
        
        mtype <- typeof(x)
        
        result <- switch(mtype,
                         integer = {cpp_vclMatrix_norm(x@address, type, 4L)},
                         float = {cpp_vclMatrix_norm(x@address, type, 6L)},
                         double = {cpp_vclMatrix_norm(x@address, type, 8L)},
                         stop("type not recognized")
        )
        
        return(result)
    }
)

#' @rdname norm-methods
#' @export
setMethod("norm", signature(x = "vclMatrix", type = "missing"),
          function(x, type){
              
              mtype <- typeof(x)
              type <- "O"
              
              result <- switch(mtype,
                               integer = {cpp_vclMatrix_norm(x@address, type, 4L)},
                               float = {cpp_vclMatrix_norm(x@address, type, 6L)},
                               double = {cpp_vclMatrix_norm(x@address, type, 8L)},
                               stop("type not recognized")
              )
              
              return(result)
          }
)

#' @rdname norm-methods
#' @export
setMethod("norm", signature(x = "gpuMatrix", type = "character"),
    function(x, type){
        
        mtype <- typeof(x)
        
        result <- switch(mtype,
                         integer = {cpp_gpuMatrix_norm(x@address, type, 4L)},
                         float = {cpp_gpuMatrix_norm(x@address, type, 6L)},
                         double = {cpp_gpuMatrix_norm(x@address, type, 8L)},
                         stop("type not recognized")
        )
        
        return(result)
    }
)

#' @rdname norm-methods
#' @export
setMethod("norm", signature(x = "gpuMatrix", type = "missing"),
          function(x, type){
              
              mtype <- typeof(x)
              type <- "O"
              
              result <- switch(mtype,
                               integer = {cpp_gpuMatrix_norm(x@address, type, 4L)},
                               float = {cpp_gpuMatrix_norm(x@address, type, 6L)},
                               double = {cpp_gpuMatrix_norm(x@address, type, 8L)},
                               stop("type not recognized")
              )
              
              return(result)
          }
)

#' @rdname norm-methods
#' @export
setMethod("norm", signature(x = "ANY", type = "missing"),
          function(x, type){
              base::norm(x, type = "O")
          }
)

#' @rdname norm-methods
#' @export
setMethod("norm", signature(x = "ANY", type = "character"),
          function(x, type){
              base::norm(x, type = type)
          }
)
