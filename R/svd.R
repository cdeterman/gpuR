
# setOldClass("svd")

# create S3 generic
#' @title Singular Value Decomposition of a gpuR matrix
#' @description Compute the singular-value decomposition of a gpuR matrix
#' @param x A gpuR matrix
#' @param nu ignored
#' @param nv ignored
#' @param LINPACK ignored
#' @return The SVD decomposition of the matrix.  The returned value is a list
#' with the following components:
#' \itemize{
#' \item{d} a vector containing the singular values of \code{x}
#' \item{u} a matrix whose columns contain the left singular vectors of 
#' \code{x}.
#' \item{v} a matrix whose columns contain the right singular vectors of 
#' \code{x}.
#' }
#' @note This an S3 generic of \link[base]{svd}.  The default continues
#' to point to the default base function.
#' @author Charles Determan Jr.
#' @rdname svd-methods
#' @seealso \link[base]{svd}
#' @export
svd <- function(x, nu, nv, LINPACK) UseMethod("svd")

# add base method
#' @export
svd.default <- base::svd


# GPU Singular Value Decomposition

#' @rdname svd-methods
#' @export
svd.vclMatrix <- 
          function(x, nu, nv, LINPACK){
              
              if(ncol(x) != nrow(x)){
                  stop("non-square matrix not currently supported for 'svd'")
              }
              
              type <- typeof(x)
              
              D <- vclVector(length = as.integer(min(nrow(x), ncol(x))), type = type, ctx_id=x@.context_index)
              U <- vclMatrix(0, ncol = nrow(x), nrow = nrow(x), type = type, ctx_id=x@.context_index)
              V <- vclMatrix(0, ncol = ncol(x), nrow = ncol(x), type = type, ctx_id=x@.context_index)
              
              switch(type,
                     integer = {cpp_vclMatrix_svd(x@address, D@address, U@address, V@address, 4L, ctx_id = x@.context_index - 1)},
                     float = {cpp_vclMatrix_svd(x@address, D@address, U@address, V@address, 6L, ctx_id = x@.context_index - 1)},
                     double = {cpp_vclMatrix_svd(x@address, D@address, U@address, V@address, 8L, ctx_id = x@.context_index - 1)},
                     stop("type not recognized")
              )
              
              return(list(d = D, u = U, v = V))
          }
# )


#' @rdname svd-methods
#' @export
svd.gpuMatrix <- 
          function(x, nu, nv, LINPACK){
              
              if(ncol(x) != nrow(x)){
                  stop("non-square matrix not currently supported for 'svd'")
              }
              
              type <- typeof(x)
              
              D <- gpuVector(length = as.integer(min(nrow(x), ncol(x))), type = type, ctx_id=x@.context_index)
              U <- gpuMatrix(0, ncol = nrow(x), nrow = nrow(x), type = type, ctx_id=x@.context_index)
              V <- gpuMatrix(0, ncol = ncol(x), nrow = ncol(x), type = type, ctx_id=x@.context_index)
              
              switch(type,
                     integer = {cpp_gpuMatrix_svd(x@address, D@address, U@address, V@address, 4L, ctx_id = x@.context_index - 1)},
                     float = {cpp_gpuMatrix_svd(x@address, D@address, U@address, V@address, 6L, ctx_id = x@.context_index - 1)},
                     double = {cpp_gpuMatrix_svd(x@address, D@address, U@address, V@address, 8L, ctx_id = x@.context_index - 1)},
                     stop("type not recognized")
              )
              
              return(list(d = D, u = U, v = V))
          }
# )
