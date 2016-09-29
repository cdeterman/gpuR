

#' @useDynLib gpuR
#' @importFrom Rcpp evalCpp


### gpuMatrix Wrappers ###


#' @importFrom Rcpp evalCpp


# GPU axpy wrapper
gpu_Mat_axpy <- function(alpha, A, B){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    nrA = nrow(A)
    ncA = ncol(A)
    nrB = nrow(B)
    ncB = ncol(B)
    
    type <- typeof(A)
    
    Z <- gpuMatrix(nrow=nrB, ncol=ncA, type=type, ctx_id = A@.context_index)
    if(!missing(B))
    {
        if(length(B) != length(A)) stop("Lengths of matrices must match")
        Z <- deepcopy(B)
    }
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               # cpp_gpuMatrix_iaxpy(alpha, 
               #                     A@address,
               #                     Z@address, 
               #                     kernel,
               #                     device_flag)
               cpp_gpuMatrix_axpy(alpha, 
                                  A@address, 
                                  Z@address, 
                                  4L,
                                  A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_axpy(alpha, 
                                       A@address, 
                                       Z@address, 
                                       6L,
                                       A@.context_index - 1)
           },
           double = {cpp_gpuMatrix_axpy(alpha, 
                                        A@address,
                                        Z@address,
                                        8L,
                                        A@.context_index - 1)
           },
           stop("type not recognized")
    )
    
    return(Z)
}

# GPU axpy wrapper
gpuMatrix_unary_axpy <- function(A){
    
    type = typeof(A)
    
    Z <- deepcopy(A)
    
    switch(type,
           integer = {
               cpp_gpuMatrix_unary_axpy(Z@address, 
                                        4L,
                                        A@.context_index - 1)
           },
           float = {
               cpp_gpuMatrix_unary_axpy(Z@address, 
                                        6L,
                                        A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_unary_axpy(Z@address,
                                        8L,
                                        A@.context_index - 1)
           },
           stop("type not recognized")
    )
    
    return(Z)
}

# GPU Matrix Multiplication
gpu_Mat_mult <- function(A, B){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(B), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               
               file <- system.file("CL", "basic_gemm.cl", package = "gpuR")
               
               if(!file_test("-f", file)){
                   stop("kernel file does not exist")
               }
               kernel <- readChar(file, file.info(file)$size)
               
               maxWorkGroupSize <- 
                   switch(deviceType(C@.platform_index, C@.device_index),
                          "gpu" = gpuInfo(C@.platform_index, C@.device_index)$maxWorkGroupSize,
                          "cpu" = cpuInfo(C@.platform_index, C@.device_index)$maxWorkGroupSize,
                          stop("unrecognized device type")
                   )
               
               cpp_gpuMatrix_custom_igemm(A@address,
                                          FALSE,
                                          B@address,
                                          FALSE,
                                          C@address,
                                          FALSE,
                                          kernel,
                                          sqrt(maxWorkGroupSize),
                                          A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_gemm(A@address,
                                       B@address,
                                       C@address,
                                       6L,
                                       A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_gemm(A@address,
                                  B@address,
                                  C@address,
                                  8L,
                                  A@.context_index - 1)
           },
           stop("type not recognized")
    )
    
    return(C)
}

# GPU Element-Wise Multiplication
gpuMatElemMult <- function(A, B){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    if(!all(dim(A) == dim(B))){
        stop("matrices not conformable")
    }
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_prod(A@address,
                                       B@address,
                                       C@address,
                                       4L,
                                       A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_elem_prod(A@address,
                                            B@address,
                                            C@address,
                                            6L,
                                            A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_elem_prod(A@address,
                                       B@address,
                                       C@address,
                                       8L,
                                       A@.context_index - 1)
           },
           {
               stop("type not recognized")
           })
    return(C)
}

# GPU Scalar Element-Wise Multiplication
gpuMatScalarMult <- function(A, B){
    
    type <- typeof(A)
    
    C <- deepcopy(A)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_scalar_prod(C@address,
                                         B,
                                         4L,
                                         A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_scalar_prod(C@address,
                                              B,
                                              6L,
                                              A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_scalar_prod(C@address,
                                         B,
                                         8L,
                                         A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Division
gpuMatElemDiv <- function(A, B){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    if(!all(dim(A) == dim(B))){
        stop("matrices not conformable")
    }
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_div(A@address,
                                      B@address,
                                      C@address,
                                      4L,
                                      A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_elem_div(A@address,
                                           B@address,
                                           C@address,
                                           6L,
                                           A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_elem_div(A@address,
                                      B@address,
                                      C@address,
                                      8L,
                                      A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Scalar Element-Wise Division
gpuMatScalarDiv <- function(A, B){
    
    type <- typeof(A)
    
    C <- deepcopy(A)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_scalar_div(C@address,
                                        B,
                                        4L,
                                        A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_scalar_div(C@address,
                                             B,
                                             6L,
                                             A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_scalar_div(C@address,
                                        B,
                                        8L,
                                        A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Power
gpuMatElemPow <- function(A, B){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    if(!all(dim(A) == dim(B))){
        stop("matrices not conformable")
    }
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_pow(A@address,
                                      B@address,
                                      C@address,
                                      4L,
                                      A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_elem_pow(A@address,
                                           B@address,
                                           C@address,
                                           6L,
                                           A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_elem_pow(A@address,
                                      B@address,
                                      C@address,
                                      8L,
                                      A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Power
gpuMatScalarPow <- function(A, B){
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_scalar_pow(A@address,
                                        B,
                                        C@address,
                                        4L,
                                        A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_scalar_pow(A@address,
                                             B,
                                             C@address,
                                             6L,
                                             A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_scalar_pow(A@address,
                                        B,
                                        C@address,
                                        8L,
                                        A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Sine
gpuMatElemSin <- function(A){
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_sin(A@address,
                                      C@address,
                                      4L,
                                      A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_elem_sin(A@address,
                                           C@address,
                                           6L,
                                           A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_elem_sin(A@address,
                                      C@address,
                                      8L,
                                      A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Arc Sine
gpuMatElemArcSin <- function(A){
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_asin(A@address,
                                       C@address,
                                       4L,
                                       A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_elem_asin(A@address,
                                            C@address,
                                            6L,
                                            A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_elem_asin(A@address,
                                       C@address,
                                       8L,
                                       A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Hyperbolic Sine
gpuMatElemHypSin <- function(A){
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_sinh(A@address,
                                       C@address,
                                       4L,
                                       A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_elem_sinh(A@address,
                                            C@address,
                                            6L,
                                            A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_elem_sinh(A@address,
                                       C@address,
                                       8L,
                                       A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Cos
gpuMatElemCos <- function(A){
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_cos(A@address,
                                      C@address,
                                      4L,
                                      A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_elem_cos(A@address,
                                           C@address,
                                           6L,
                                           A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_elem_cos(A@address,
                                      C@address,
                                      8L,
                                      A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Arc Cos
gpuMatElemArcCos <- function(A){
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_acos(A@address,
                                       C@address,
                                       4L,
                                       A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_elem_acos(A@address,
                                            C@address,
                                            6L,
                                            A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_elem_acos(A@address,
                                       C@address,
                                       8L,
                                       A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Hyperbolic Cos
gpuMatElemHypCos <- function(A){
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_cosh(A@address,
                                       C@address,
                                       4L,
                                       A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_elem_cosh(A@address,
                                            C@address,
                                            6L,
                                            A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_elem_cosh(A@address,
                                       C@address,
                                       8L,
                                       A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Tan
gpuMatElemTan <- function(A){
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_tan(A@address,
                                      C@address,
                                      4L,
                                      A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_elem_tan(A@address,
                                           C@address,
                                           6L,
                                           A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_elem_tan(A@address,
                                      C@address,
                                      8L,
                                      A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Arc Tan
gpuMatElemArcTan <- function(A){
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_atan(A@address,
                                       C@address,
                                       4L,
                                       A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_elem_atan(A@address,
                                            C@address,
                                            6L,
                                            A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_elem_atan(A@address,
                                       C@address,
                                       8L,
                                       A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Hyperbolic Tan
gpuMatElemHypTan <- function(A){
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_tanh(A@address,
                                       C@address,
                                       4L,
                                       A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_elem_tanh(A@address,
                                            C@address,
                                            6L,
                                            A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_elem_tanh(A@address,
                                       C@address,
                                       8L,
                                       A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Natural Log
gpuMatElemLog <- function(A){
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_log(A@address,
                                      C@address,
                                      4L,
                                      A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_elem_log(A@address,
                                           C@address,
                                           6L,
                                           A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_elem_log(A@address,
                                      C@address,
                                      8L,
                                      A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Log Base
gpuMatElemLogBase <- function(A, base){
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_log_base(A@address,
                                           C@address,
                                           base,
                                           4L,
                                           A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_elem_log_base(A@address,
                                                C@address,
                                                base,
                                                6L,
                                                A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_elem_log_base(A@address,
                                           C@address,
                                           base,
                                           8L,
                                           A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Base 10 Log
gpuMatElemLog10 <- function(A){
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_log10(A@address,
                                        C@address,
                                        4L,
                                        A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_elem_log10(A@address,
                                             C@address,
                                             6L,
                                             A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_elem_log10(A@address,
                                        C@address,
                                        8L,
                                        A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Exponential
gpuMatElemExp <- function(A){
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_exp(A@address,
                                      C@address,
                                      4L,
                                      A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_elem_exp(A@address,
                                           C@address,
                                           6L,
                                           A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_elem_exp(A@address,
                                      C@address,
                                      8L,
                                      A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU colSums
gpu_colSums <- function(A){
    
    type <- typeof(A)
    
    sums <- gpuVector(length = ncol(A), type = type, ctx_id = A@.context_index)
    
    switch(type,
           "integer" = {
               # stop("integer type not currently implemented")
               cpp_gpuMatrix_colsum(A@address, 
                                    sums@address, 
                                    4L,
                                    A@.context_index - 1)
           },
           "float" = {
               cpp_gpuMatrix_colsum(A@address, 
                                    sums@address, 
                                    6L,
                                    A@.context_index - 1)
           },
           "double" = {
               cpp_gpuMatrix_colsum(A@address, 
                                    sums@address, 
                                    8L,
                                    A@.context_index - 1)
           },
           stop("unsupported matrix type")
    )
    
    return(sums)
}

# GPU rowSums
gpu_rowSums <- function(A){
    
    type <- typeof(A)
    
    sums <- gpuVector(length = nrow(A), type = type, ctx_id = A@.context_index)
    
    switch(type,
           "integer" = {
               # stop("integer type not currently implemented")
               cpp_gpuMatrix_rowsum(
                   A@address, 
                   sums@address,
                   4L,
                   A@.context_index - 1)
           },
           "float" = {
               cpp_gpuMatrix_rowsum(
                   A@address, 
                   sums@address,
                   6L,
                   A@.context_index - 1)
           },
           "double" = {
               cpp_gpuMatrix_rowsum(
                   A@address, 
                   sums@address,
                   8L,
                   A@.context_index - 1)
           },
           stop("unsupported matrix type")
    )
    
    return(sums)
}

# GPU colMeans
gpu_colMeans <- function(A){
    
    type <- typeof(A)
    
    sums <- gpuVector(length = ncol(A), type = type, ctx_id = A@.context_index)
    
    switch(type,
           "integer" = {
               # stop("integer type not currently implemented")
               cpp_gpuMatrix_colmean(A@address, 
                                     sums@address, 
                                     4L,
                                     A@.context_index - 1)
           },
           "float" = cpp_gpuMatrix_colmean(A@address, 
                                           sums@address, 
                                           6L,
                                           A@.context_index - 1),
           "double" = cpp_gpuMatrix_colmean(A@address, 
                                            sums@address, 
                                            8L,
                                            A@.context_index - 1)
    )
    
    return(sums)
}

# GPU rowMeans
gpu_rowMeans <- function(A){
    
    type <- typeof(A)
    
    sums <- gpuVector(length = nrow(A), type = type, ctx_id = A@.context_index)
    
    switch(type,
           "integer" = {
               # stop("integer type not currently implemented")
               cpp_gpuMatrix_rowmean(A@address, 
                                     sums@address, 
                                     4L,
                                     A@.context_index - 1)
           },
           "float" = cpp_gpuMatrix_rowmean(A@address, 
                                           sums@address, 
                                           6L,
                                           A@.context_index - 1),
           "double" = cpp_gpuMatrix_rowmean(A@address, 
                                            sums@address, 
                                            8L,
                                            A@.context_index - 1)
    )
    
    return(sums)
}

# GPU Pearson Covariance
gpu_pmcc <- function(A){
    
    type <- typeof(A)
    
    B <- gpuMatrix(nrow = ncol(A), ncol = ncol(A), type = type, ctx_id = A@.context_index)
    
    switch(type,
           "integer" = {
               stop("integer type not currently implemented")
               # cpp_gpuMatrix_pmcc(A@address, 
               #                    B@address, 
               #                    4L,
               #                    A@.context_index - 1)
           },
           "float" = cpp_gpuMatrix_pmcc(A@address, 
                                        B@address, 
                                        6L,
                                        A@.context_index - 1),
           "double" = cpp_gpuMatrix_pmcc(A@address, 
                                         B@address, 
                                         8L,
                                         A@.context_index - 1)
    )
    
    return(B)
}

# GPU crossprod
gpu_crossprod <- function(X, Y){
    
    if(nrow(X) != nrow(Y)){
        stop("matrices non-conformable")
    }
    
    type <- typeof(X)
    
    Z <- gpuMatrix(nrow = ncol(X), ncol = ncol(Y), type = type, ctx_id = X@.context_index)
    
    switch(type,
           "integer" = {
               stop("integer type not currently implemented")
               # cpp_gpuMatrix_crossprod(X@address, 
               #                         Y@address, 
               #                         Z@address,
               #                         4L,
               #                         X@.context_index - 1)
           },
           "float" = {
               cpp_gpuMatrix_crossprod(X@address, 
                                       Y@address, 
                                       Z@address,
                                       6L,
                                       X@.context_index - 1)
           },
           "double" = {
               cpp_gpuMatrix_crossprod(X@address, 
                                       Y@address, 
                                       Z@address, 
                                       8L,
                                       X@.context_index - 1)
           },
           stop("Unsupported type")
    )
    
    return(Z)
}

# GPU tcrossprod
gpu_tcrossprod <- function(X, Y){
    
    if(ncol(X) != ncol(Y)){
        stop("matrices non-conformable")
    }
    
    type <- typeof(X)
    
    Z <- gpuMatrix(nrow = nrow(X), ncol = nrow(Y), type = type, ctx_id = X@.context_index)
    
    switch(type,
           "integer" = {
               stop("integer type not currently implemented")
               # cpp_gpuMatrix_tcrossprod(X@address, 
               #                          Y@address, 
               #                          Z@address, 
               #                          4L,
               #                          X@.context_index - 1)
           },
           "float" = {
               cpp_gpuMatrix_tcrossprod(X@address, 
                                        Y@address, 
                                        Z@address, 
                                        6L,
                                        X@.context_index - 1)
           },
           "double" = {
               cpp_gpuMatrix_tcrossprod(X@address,
                                        Y@address, 
                                        Z@address, 
                                        8L,
                                        X@.context_index - 1)
           },
           stop("unsupported type")
    )
    
    return(Z)
}

# GPU Euclidean Distance
gpuMatrix_euclidean <- function(A, D, diag, upper, p, squareDist){
    
    type <- typeof(D)
    
    switch(type,
           "integer" = {
               stop("integer method not currently implemented")
               
               # cpp_gpuMatrix_eucl(A@address, 
               #                    D@address, 
               #                    squareDist,
               #                    4L,
               #                    A@.context_index - 1)
               },
           "float" = cpp_gpuMatrix_eucl(A@address, 
                                        D@address, 
                                        squareDist,
                                        6L,
                                        A@.context_index - 1),
           "double" = cpp_gpuMatrix_eucl(A@address, 
                                         D@address,
                                         squareDist,
                                         8L,
                                         A@.context_index - 1),
           stop("Unsupported matrix type")
    )
    
    invisible(D)
}

# GPU Pairwise Euclidean Distance
gpuMatrix_peuclidean <- function(A, B, D, squareDist){
    
    type <- typeof(D)
    
    switch(type,
           "integer" = {
               stop("integer method not currently implemented")
               # cpp_gpuMatrix_peucl(A@address,
               #                     B@address,
               #                     D@address,
               #                     squareDist,
               #                     4L,
               #                     A@.context_index - 1)
           },
           "float" = cpp_gpuMatrix_peucl(A@address,
                                         B@address,
                                         D@address, 
                                         squareDist, 
                                         6L,
                                         A@.context_index - 1),
           "double" = cpp_gpuMatrix_peucl(A@address, 
                                          B@address,
                                          D@address,
                                          squareDist,
                                          8L,
                                          A@.context_index - 1),
           stop("Unsupported matrix type")
    )
    
    invisible(D)
}

# GPU Element-Wise Absolute Value
gpuMatElemAbs <- function(A){
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_abs(A@address,
                                      C@address,
                                      4L,
                                      A@.context_index - 1)
           },
           float = {cpp_gpuMatrix_elem_abs(A@address,
                                           C@address,
                                           6L,
                                           A@.context_index - 1)
           },
           double = {
               cpp_gpuMatrix_elem_abs(A@address,
                                      C@address,
                                      8L,
                                      A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Matrix maximum
gpuMatrix_max <- function(A){
    
    type <- typeof(A)
    
    C <- switch(type,
                integer = {cpp_gpuMatrix_max(A@address, 4L)},
                float = {cpp_gpuMatrix_max(A@address, 6L)},
                double = {cpp_gpuMatrix_max(A@address, 8L)},
                stop("type not recognized")
    )
    return(C)
}

# GPU Matrix minimum
gpuMatrix_min <- function(A){
    
    type <- typeof(A)
    
    C <- switch(type,
                integer = {cpp_gpuMatrix_min(A@address, 4L)},
                float = {cpp_gpuMatrix_min(A@address, 6L)},
                double = {cpp_gpuMatrix_min(A@address, 8L)},
                stop("type not recognized")
    )
    return(C)
}

# GPU Matrix transpose
gpuMatrix_t <- function(A){
    
    type <- typeof(A)
    
    init = ifelse(type == "integer", 0L, 0)
    
    B <- gpuMatrix(init, ncol = nrow(A), nrow = ncol(A), type = type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {cpp_gpuMatrix_transpose(A@address, B@address, 4L,
                                              A@.context_index - 1)},
           float = {cpp_gpuMatrix_transpose(A@address, B@address,  6L,
                                            A@.context_index - 1)},
           double = {cpp_gpuMatrix_transpose(A@address, B@address,  8L,
                                             A@.context_index - 1)},
           stop("type not recognized")
    )
    
    return(B)
}

