
#' @title Compute the Norm of a Matrix
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

