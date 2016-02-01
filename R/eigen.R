#' @import methods

setOldClass("eigen")

#' @title gpuMatrix Eigen Decomposition
#' @description Computes the eigenvalues and eigenvectors for gpuMatrix objects.
#' @param x A gpuMatrix object
#' @param symmetric logical indication if matrix is assumed to be symmetric.
#' If not specified or FALSE, the matrix is inspected for symmetry
#' @param only.values if TRUE, returns only eigenvalues (internals still
#' currently calculate both regardless)
#' @param EISPACK logical. Defunct and ignored
#' @details This function currently implements the \code{qr_method} function
#' from the ViennaCL library.  As such, non-symmetric matrices are not 
#' supported given that OpenCL does not have a 'complex' data type.
#' 
#' Neither the eigenvalues nor the eigenvectors are sorted as done in the
#' base R eigen method.  
#' 
#' @note The sign's may be different on some of the eigenvector elements.
#' As noted in the base eigen documentation:
#' 
#' Recall that the eigenvectors are only defined up to a
#' constant: even when the length is specified they are still
#' only defined up to a scalar of modulus one (the sign for real
#' matrices).
#' 
#' Therefore, although the signs may be different, the results are
#' functionally equivalent
#' @return \item{values}{A \code{gpuVector} containing the unsorted eigenvalues 
#' of x.}
#' @return \item{vectors}{A \code{gpuMatrix} containing the unsorted 
#' eigenvectors of x}
#' @rdname eigen-gpuMatrix
#' @aliases eigen,vclMatrix
#' @export
setMethod("eigen", signature(x="gpuMatrix"),
          function(x, symmetric, only.values = FALSE, EISPACK = FALSE)
          {
              device_flag <- 
                  switch(options("gpuR.default.device.type")$gpuR.default.device.type,
                         "cpu" = 1, 
                         "gpu" = 0,
                         stop("unrecognized default device option"
                         )
                  )
              
              if( missing(symmetric) | is.null(symmetric) | !symmetric){
                  stop("Non-symmetric matrices not currently supported")
              }
              
#               if(missing(symmetric)){
#                   symmetric = FALSE
#               }
              
              if(ncol(x) != nrow(x)){
                  stop("non-square matrix in 'eigen'")
              }
              
              type = typeof(x)
              
              if( type == "integer"){
                  stop("Integer type not currently supported")
              }
              
              Q <- gpuMatrix(nrow=nrow(x), ncol=ncol(x), type=type)
              V <- gpuVector(length=as.integer(nrow(x)), type=type)
              
              
              switch(type,
                     "float" = cpp_gpu_eigen(x@address, 
                                             Q@address, 
                                             V@address,
                                             symmetric,
                                             6L,
                                             device_flag),
                     "double" = cpp_gpu_eigen(x@address,
                                              Q@address, 
                                              V@address, 
                                              symmetric,
                                              8L,
                                              device_flag),
                     stop("type not currently supported")
                     )
              
              if(only.values){
                  out <- list(values = V)
              }else{
                  out <- list(values = V, vectors = Q)
              }
              return(out)
          },
          valueClass = "list"
)

#' @rdname eigen-gpuMatrix
#' @aliases eigen,vclMatrix
#' @export
setMethod("eigen", signature(x="vclMatrix"),
          function(x, symmetric, only.values = FALSE, EISPACK = FALSE)
          {
              if(is(x, "vclMatrixBlock")){
                  stop("vclMatrixBlock not currently supported")
              }
              
              device_flag <- 
                  switch(options("gpuR.default.device.type")$gpuR.default.device.type,
                         "cpu" = 1, 
                         "gpu" = 0,
                         stop("unrecognized default device option"
                         )
                  )
              
              if( missing(symmetric) | is.null(symmetric) | !symmetric){
                  stop("Non-symmetric matrices not currently supported")
              }
              
              #               if(missing(symmetric)){
              #                   symmetric = FALSE
              #               }
              
              if(ncol(x) != nrow(x)){
                  stop("non-square matrix in 'eigen'")
              }
              
              type = typeof(x)
              
              if( type == "integer"){
                  stop("Integer type not currently supported")
              }
              
              Q <- vclMatrix(nrow=nrow(x), ncol=ncol(x), type=type)
              V <- vclVector(length=as.integer(nrow(x)), type=type)
              
              # possible a way to have only values calculated on GPU?
              
              switch(type,
                     "float" = cpp_vcl_eigen(x@address, 
                                             Q@address, 
                                             V@address,
                                             symmetric,
                                             6L,
                                             device_flag),
                     "double" = cpp_vcl_eigen(x@address,
                                              Q@address, 
                                              V@address, 
                                              symmetric,
                                              8L,
                                              device_flag),
                     stop("type not currently supported")
              )
              
              if(only.values){
                  out <- list(values = V)
              }else{
                  out <- list(values = V, vectors = Q)
              }
              return(out)
          },
          valueClass = "list"
)
