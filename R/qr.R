

#' @title The QR Decomposition of a gpuR matrix
#' @description \code{qr} computes the QR decomposition of a gpuR matrix
#' @param x A gpuR matrix
#' @param ... further arguments passed to or from other methods
#' @param inplace Logical indicating if operations performed inplace
#' @return A \code{list} containing the QR decomposition of the matrix of class
#' \code{gpuQR}.  The returned value is a list with the following components:
#' \itemize{
#' \item{qr} a matrix with the same dimensions as \code{x}.
#' \item{betas} vector of numeric values containing additional information of 
#' \code{qr} for extracting \code{Q} and \code{R} matrices.
#' }
#' @note This an S3 generic of \link[base]{qr}.  The default continues
#' to point to the default base function.
#' 
#' Furthermore, the list returned does not contain the exact same elements
#' as \link[base]{qr}.  The matrix storage format applied herein doesn't match
#' the base compact form.  The method also doesn't return \code{qraux}, 
#' \code{rank}, or \code{pivot} but instead returns \code{betas}
#' @author Charles Determan Jr.
#' @rdname qr-methods
#' @seealso \link[base]{qr}
#' @aliases qr.gpuR
#' @export
qr.gpuMatrix <-
    function(x, ..., inplace = FALSE)
    {
        type = typeof(x)
        
        if(nrow(x) != ncol(x)){
            stop("non-square matrix not currently supported for 'qr'")
        }
        
        if( type == "integer"){
            stop("Integer type not currently supported")
        }
        
        if(inplace){
            z <- x
        }else{
            z <- deepcopy(x)
        }
        
        betas <- switch(type,
                        "float" = cpp_gpuR_qr(z@address,
                                              FALSE,
                                              6L,
                                              z@.context_index - 1),
                        "double" = cpp_gpuR_qr(z@address,
                                               FALSE,
                                               8L,
                                               z@.context_index - 1),
                        stop("type not currently supported")
        )
        
        out <- list(qr = z, betas = betas)
        class(out) <- "gpuQR"
        
        return(out)
    }


#' @rdname qr-methods
#' @export
qr.vclMatrix <-
          function(x, ..., inplace = FALSE)
          {
              type = typeof(x)

              if(nrow(x) != ncol(x)){
                  stop("non-square matrix not currently supported for 'qr'")
              }

              if( type == "integer"){
                  stop("Integer type not currently supported")
              }

              if(inplace){
                  z <- x
              }else{
                  z <- deepcopy(x)
              }
              
              betas <- switch(type,
                              "float" = cpp_gpuR_qr(z@address,
                                                    TRUE,
                                                    6L,
                                                    x@.context_index - 1),
                              "double" = cpp_gpuR_qr(z@address,
                                                     TRUE,
                                                     8L,
                                                     x@.context_index - 1),
                              stop("type not currently supported")
              )

              out <- list(qr = z, betas = betas)
              class(out) <- "gpuQR"
              
              return(out)
          }

#' @title Reconstruct the Q or R Matrices from a gpuQR Object
#' @description Returns the components of the QR decomposition.
#' @param qr \code{gpuQR} object
#' @param complete not currently used
#' @return \code{qr.Q} returns all of \code{Q}
#' \code{qr.R} returns all of \code{R}
#' @author Charles Determan Jr.
#' @rdname qr.R-methods
#' @seealso \link[base]{qr.R}, \link[base]{qr.Q}
#' @export
setMethod("qr.R", signature(qr = "gpuQR"),
          function(qr, complete = FALSE){
              
              type <- typeof(qr$qr)
              
              isVCL <- if(inherits(qr$qr, "vclMatrix")) TRUE else FALSE
              
              if(isVCL){
                  Q <- vclMatrix(nrow = nrow(qr$qr), ncol = ncol(qr$qr), type = type)
                  R <- vclMatrix(nrow = nrow(qr$qr), ncol = nrow(qr$qr), type = type)        
              }else{
                  Q <- vclMatrix(nrow = nrow(qr$qr), ncol = ncol(qr$qr), type = type)
                  R <- gpuMatrix(nrow = nrow(qr$qr), ncol = nrow(qr$qr), type = type)
              }
              
              
              switch(type,
                     "float" = cpp_recover_qr(qr$qr@address,
                                              isVCL,
                                              Q@address,
                                              inherits(Q, "vclMatrix"),
                                              R@address,
                                              inherits(R, "vclMatrix"),
                                              qr$betas,
                                              6L,
                                              qr$qr@.context_index - 1),
                     "double" = cpp_recover_qr(qr$qr@address,
                                               isVCL,
                                               Q@address,
                                               inherits(Q, "vclMatrix"),
                                               R@address,
                                               inherits(R, "vclMatrix"),
                                               qr$betas,
                                               8L,
                                               qr$qr@.context_index - 1),
                     stop("type not currently supported")
              )
              
              return(R)
          }) 


#' @rdname qr.R-methods
#' @export
setMethod("qr.Q", signature(qr = "gpuQR"),
          function(qr, complete = FALSE){
              
              type <- typeof(qr$qr)
              
              isVCL <- inherits(qr$qr, "vclMatrix")
              
              if(isVCL){
                  Q <- vclMatrix(nrow = nrow(qr$qr), ncol = ncol(qr$qr), type = type)
                  R <- vclMatrix(nrow = nrow(qr$qr), ncol = nrow(qr$qr), type = type)    
              }else{
                  Q <- gpuMatrix(nrow = nrow(qr$qr), ncol = ncol(qr$qr), type = type)
                  R <- vclMatrix(nrow = nrow(qr$qr), ncol = nrow(qr$qr), type = type)
              }
              
              # print(isVCL)
              # stop("stopping")
              
              switch(type,
                     "float" = cpp_recover_qr(qr$qr@address,
                                              isVCL,
                                              Q@address,
                                              inherits(Q, "vclMatrix"),
                                              R@address,
                                              inherits(R, "vclMatrix"),
                                              qr$betas,
                                              6L,
                                              qr$qr@.context_index - 1),
                     "double" = cpp_recover_qr(qr$qr@address,
                                               isVCL,
                                               Q@address,
                                               inherits(Q, "vclMatrix"),
                                               R@address,
                                               inherits(R, "vclMatrix"),
                                               qr$betas,
                                               8L,
                                               qr$qr@.context_index - 1),
                     stop("type not currently supported")
              )
              
              return(Q)
          }) 
