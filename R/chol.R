
# gpu cholesky decomposition

#' @title Cholesky Decomposition of a gpuR matrix
#' @description Compute the Choleski factorization of a real symmetric 
#' positive-definite square matrix.
#' @param x A symmetric, positive-definite gpuR matrix object.
#' @param ... arguments to be passed to or from methods
#' @return Default - the upper triangular factor of the Choleski decomposition,
#' i.e. the matrix \emph{R} such that \emph{R'R} = x.
#' @note This an S3 generic of \link[base]{chol}.  The default continues
#' to point to the default base function.
#' 
#' No pivoting is used.
#' 
#' The argument \code{upper} is additionally accepted representing a boolean 
#' which will indicate if the upper or lower (\code{FALSE}) triangle
#' should be solved.
#' 
#' @author Charles Determan Jr.
#' @rdname chol-methods
#' @seealso \link[base]{chol}
#' @export
chol.vclMatrix <- function(x, ...){
    
    d <- dim(x)
    if(d[1] != d[2]){
        stop("'x' must be a square matrix")
    }
    
    type <- typeof(x)
    
    myargs <- as.list(sys.call())
    
    if(!'upper' %in% names(myargs)){
        upper = TRUE
    }else{
        assert_is_a_bool(upper)    
    }
    
    # convert to integer for OpenCL as can't pass bool to kernel :P
    upper <- as.integer(upper)
    
    # B <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    B <- deepcopy(x)
    
    switch(type,
           integer = {
               stop("OpenCL integer Cholesky not currently
                    supported for viennacl matrices")
           },
           float = {
               file <- system.file("CL", "fcholesky.cl", package = "gpuR")
               
               if(!file_test("-f", file)){
                   stop("kernel file does not exist")
               }
               kernel <- readChar(file, file.info(file)$size)
               
               maxWorkGroupSize <- 
                   switch(deviceType(B@.platform_index, B@.device_index),
                          "gpu" = gpuInfo(B@.platform_index, B@.device_index)$maxWorkGroupSize,
                          "cpu" = cpuInfo(B@.platform_index, B@.device_index)$maxWorkGroupSize,
                          stop("unrecognized device type")
                   )
               
               cpp_vclMatrix_custom_chol(B@address,
                                         TRUE,
                                         upper,
                                         kernel,
                                         sqrt(maxWorkGroupSize),
                                         6L,
                                         B@.context_index - 1)
           },
           double = {
               file <- system.file("CL", "dcholesky.cl", package = "gpuR")
               
               if(!file_test("-f", file)){
                   stop("kernel file does not exist")
               }
               kernel <- readChar(file, file.info(file)$size)
               
               maxWorkGroupSize <- 
                   switch(deviceType(B@.platform_index, B@.device_index),
                          "gpu" = gpuInfo(B@.platform_index, B@.device_index)$maxWorkGroupSize,
                          "cpu" = cpuInfo(B@.platform_index, B@.device_index)$maxWorkGroupSize,
                          stop("unrecognized device type")
                   )
               
               cpp_vclMatrix_custom_chol(B@address,
                                         TRUE,
                                         upper,
                                         kernel,
                                         sqrt(maxWorkGroupSize),
                                         8L,
                                         B@.context_index - 1)
           },
           stop("type not recognized")
    )
    
    return(B)
}


#' @rdname chol-methods
#' @export
chol.gpuMatrix <- function(x, ...){
    
    d <- dim(x)
    if(d[1] != d[2]){
        stop("'x' must be a square matrix")
    }
    
    type <- typeof(x)
    
    # this is necessary to avoid R CMD check conflicts
    myargs <- as.list(sys.call())
    
    if(!'upper' %in% names(myargs)){
        upper = TRUE
    }else{
        assert_is_a_bool(upper)    
    }
    
    # convert to integer for OpenCL as can't pass bool to kernel :P
    upper <- as.integer(upper)
    
    # B <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    B <- deepcopy(x)
    
    switch(type,
           integer = {
               stop("OpenCL integer Cholesky not currently
                    supported for viennacl matrices")
           },
           float = {
               file <- system.file("CL", "fcholesky.cl", package = "gpuR")
               
               if(!file_test("-f", file)){
                   stop("kernel file does not exist")
               }
               kernel <- readChar(file, file.info(file)$size)
               
               maxWorkGroupSize <- 
                   switch(deviceType(B@.platform_index, B@.device_index),
                          "gpu" = gpuInfo(B@.platform_index, B@.device_index)$maxWorkGroupSize,
                          "cpu" = cpuInfo(B@.platform_index, B@.device_index)$maxWorkGroupSize,
                          stop("unrecognized device type")
                   )
               
               cpp_vclMatrix_custom_chol(B@address,
                                         FALSE,
                                         upper,
                                         kernel,
                                         sqrt(maxWorkGroupSize),
                                         6L,
                                         B@.context_index - 1)
           },
           double = {
               file <- system.file("CL", "dcholesky.cl", package = "gpuR")
               
               if(!file_test("-f", file)){
                   stop("kernel file does not exist")
               }
               kernel <- readChar(file, file.info(file)$size)
               
               maxWorkGroupSize <- 
                   switch(deviceType(B@.platform_index, B@.device_index),
                          "gpu" = gpuInfo(B@.platform_index, B@.device_index)$maxWorkGroupSize,
                          "cpu" = cpuInfo(B@.platform_index, B@.device_index)$maxWorkGroupSize,
                          stop("unrecognized device type")
                   )
               
               cpp_vclMatrix_custom_chol(B@address,
                                         FALSE,
                                         upper,
                                         kernel,
                                         sqrt(maxWorkGroupSize),
                                         8L,
                                         B@.context_index - 1)
           },
           stop("type not recognized")
    )
    
    return(B)
}

