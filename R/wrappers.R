

#' @useDynLib gpuR
#' @importFrom Rcpp evalCpp


### gpuMatrix Wrappers ###


#' @importFrom Rcpp evalCpp


# GPU axpy wrapper
gpu_Mat_axpy <- function(alpha, A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    nrA = nrow(A)
    ncA = ncol(A)
    nrB = nrow(B)
    ncB = ncol(B)
    
#     pkg_path <- find.package("gpuR", .libPaths())
#     file <- file.path(pkg_path, "CL", "basic_axpy.cl")
    
    file <- system.file("CL", "basic_axpy.cl", package = "gpuR")
    
    if(!file_test("-f", file)){
        stop("kernel file does not exist")
    }
    kernel <- readChar(file, file.info(file)$size)
    
    type <- typeof(A)
    
    Z <- gpuMatrix(nrow=nrB, ncol=ncA, type=type)
    if(!missing(B))
    {
        if(length(B[]) != length(A[])) stop("Lengths of matrices must match")
        Z <- deepcopy(B)
    }
    
    switch(type,
           integer = {cpp_gpuMatrix_iaxpy(alpha, 
                                          A@address,
                                          Z@address, 
                                          kernel,
                                          device_flag)
           },
           float = {cpp_gpuMatrix_axpy(alpha, 
                                       A@address, 
                                       Z@address, 
                                       device_flag,
                                       6L)
           },
           double = {cpp_gpuMatrix_axpy(alpha, 
                                        A@address,
                                        Z@address,
                                        device_flag,
                                        8L)
           },
           stop("type not recognized")
    )
    
    return(Z)
}

# GPU axpy wrapper
gpuMatrix_unary_axpy <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type = typeof(A)
    
    Z <- deepcopy(A)
    
    switch(type,
           integer = {
               cpp_gpuMatrix_unary_axpy(Z@address, 
                                        device_flag,
                                        4L)
           },
           float = {
               cpp_gpuMatrix_unary_axpy(Z@address, 
                                       device_flag,
                                       6L)
           },
           double = {
               cpp_gpuMatrix_unary_axpy(Z@address,
                                        device_flag,
                                        8L)
           },
           stop("type not recognized")
    )
    
    return(Z)
}

# GPU Matrix Multiplication
gpu_Mat_mult <- function(A, B){
    
#     pkg_path <- find.package("gpuR", .libPaths())
#     file <- file.path(pkg_path, "CL", "basic_gemm.cl")
    
    file <- system.file("CL", "basic_gemm.cl", package = "gpuR")
    
    if(!file_test("-f", file)){
        stop("kernel file does not exist")
    }
    kernel <- readChar(file, file.info(file)$size)
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(B), type=type)
    
#     print(C[])
    
    switch(type,
           integer = {
               cpp_gpuMatrix_igemm(A@address,
                                   B@address, 
                                   C@address,
                                   kernel,
                                   device_flag)
               #                      cpp_vienna_gpuMatrix_igemm(A@address,
               #                                                        B@address,
               #                                                        C@address)
           },
           float = {cpp_gpuMatrix_gemm(A@address,
                                       B@address,
                                       C@address,
                                       device_flag,
                                       6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_gemm(A@address,
                                        B@address,
                                        C@address,
                                        device_flag,
                                        8L)
               }
           },
{
    stop("type not recognized")
})
#     rm(C)
return(C)
}

# GPU Element-Wise Multiplication
gpuMatElemMult <- function(A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    if(!all(dim(A) == dim(B))){
        stop("matrices not conformable")
    }
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuMatrix_elem_prod(A@address,
                                            B@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_elem_prod(A@address,
                                             B@address,
                                             C@address,
                                             device_flag,
                                             8L)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Scalar Element-Wise Multiplication
gpuMatScalarMult <- function(A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- deepcopy(A)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuMatrix_scalar_prod(C@address,
                                              B,
                                              device_flag,
                                              6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_scalar_prod(C@address,
                                               B,
                                               device_flag,
                                               8L)
               }
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Division
gpuMatElemDiv <- function(A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    if(!all(dim(A) == dim(B))){
        stop("matrices not conformable")
    }
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuMatrix_elem_div(A@address,
                                           B@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_elem_div(A@address,
                                            B@address,
                                            C@address,
                                            device_flag,
                                            8L)
               }
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Scalar Element-Wise Division
gpuMatScalarDiv <- function(A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- deepcopy(A)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuMatrix_scalar_div(C@address,
                                             B,
                                             device_flag,
                                             6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_scalar_div(C@address,
                                              B,
                                              device_flag,
                                              8L)
               }
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Power
gpuMatElemPow <- function(A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    if(!all(dim(A) == dim(B))){
        stop("matrices not conformable")
    }
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuMatrix_elem_pow(A@address,
                                           B@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_elem_pow(A@address,
                                            B@address,
                                            C@address,
                                            device_flag,
                                            8L)
               }
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Power
gpuMatScalarPow <- function(A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuMatrix_scalar_pow(A@address,
                                           B,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_scalar_pow(A@address,
                                            B,
                                            C@address,
                                            device_flag,
                                            8L)
               }
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Sine
gpuMatElemSin <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuMatrix_elem_sin(A@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_elem_sin(A@address,
                                            C@address,
                                            device_flag,
                                            8L)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Arc Sine
gpuMatElemArcSin <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuMatrix_elem_asin(A@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_elem_asin(A@address,
                                             C@address,
                                             device_flag,
                                             8L)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Hyperbolic Sine
gpuMatElemHypSin <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuMatrix_elem_sinh(A@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_elem_sinh(A@address,
                                             C@address,
                                             device_flag,
                                             8L)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Cos
gpuMatElemCos <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuMatrix_elem_cos(A@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_elem_cos(A@address,
                                            C@address,
                                            device_flag,
                                            8L)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Arc Cos
gpuMatElemArcCos <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuMatrix_elem_acos(A@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_elem_acos(A@address,
                                             C@address,
                                             device_flag,
                                             8L)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Hyperbolic Cos
gpuMatElemHypCos <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuMatrix_elem_cosh(A@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_elem_cosh(A@address,
                                             C@address,
                                             device_flag,
                                             8L)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Tan
gpuMatElemTan <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuMatrix_elem_tan(A@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_elem_tan(A@address,
                                            C@address,
                                            device_flag,
                                            8L)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Arc Tan
gpuMatElemArcTan <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuMatrix_elem_atan(A@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_elem_atan(A@address,
                                             C@address,
                                             device_flag,
                                             8L)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Hyperbolic Tan
gpuMatElemHypTan <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuMatrix_elem_tanh(A@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_elem_tanh(A@address,
                                             C@address,
                                             device_flag,
                                             8L)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Natural Log
gpuMatElemLog <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuMatrix_elem_log(A@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_elem_log(A@address,
                                            C@address,
                                            device_flag,
                                            8L)
               }
           },
           
           stop("type not recognized")
    )
    return(C)
}


# GPU Element-Wise Log Base
gpuMatElemLogBase <- function(A, base){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuMatrix_elem_log_base(A@address,
                                                C@address,
                                                base,
                                                device_flag,
                                                6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_elem_log_base(A@address,
                                                 C@address,
                                                 base,
                                                 device_flag,
                                                 8L)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Base 10 Log
gpuMatElemLog10 <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuMatrix_elem_log10(A@address,
                                             C@address,
                                             device_flag,
                                             6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_elem_log10(A@address,
                                              C@address,
                                              device_flag,
                                              8L)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Exponential
gpuMatElemExp <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuMatrix_elem_exp(A@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_elem_exp(A@address,
                                            C@address,
                                            device_flag,
                                            8L)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU colSums
gpu_colSums <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    if(type == "integer"){
        stop("integer type not currently implemented")
    }
    
    sums <- gpuVector(length = ncol(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = {
               cpp_gpuMatrix_colsum(A@address, 
                                    sums@address, 
                                    device_flag,
                                    6L)
               },
           "double" = {
               cpp_gpuMatrix_colsum(A@address, 
                                    sums@address, 
                                    device_flag,
                                    8L)
           },
           stop("unsupported matrix type")
    )
    
    return(sums)
}

# GPU rowSums
gpu_rowSums <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    if(type == "integer"){
        stop("integer type not currently implemented")
    }
    
    sums <- gpuVector(length = nrow(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = {
               cpp_gpuMatrix_rowsum(
                   A@address, 
                   sums@address, 
                   device_flag,
                   6L)
               },
           "double" = {
               cpp_gpuMatrix_rowsum(
                   A@address, 
                   sums@address, 
                   device_flag,
                   8L)
           },
           stop("unsupported matrix type")
    )
    
    return(sums)
}

# GPU colMeans
gpu_colMeans <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    if(type == "integer"){
        stop("integer type not currently implemented")
    }
    
    sums <- gpuVector(length = ncol(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_gpuMatrix_colmean(A@address, 
                                           sums@address, 
                                           device_flag,
                                           6L),
           "double" = cpp_gpuMatrix_colmean(A@address, 
                                            sums@address, 
                                            device_flag,
                                            8L)
    )
    
    return(sums)
}

# GPU rowMeans
gpu_rowMeans <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    if(type == "integer"){
        stop("integer type not currently implemented")
    }
    
    sums <- gpuVector(length = nrow(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_gpuMatrix_rowmean(A@address, 
                                           sums@address, 
                                           device_flag,
                                           6L),
           "double" = cpp_gpuMatrix_rowmean(A@address, 
                                            sums@address, 
                                            device_flag,
                                            8L)
    )
    
    return(sums)
}

# GPU Pearson Covariance
gpu_pmcc <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    B <- gpuMatrix(nrow = ncol(A), ncol = ncol(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_gpuMatrix_pmcc(A@address, 
                                        B@address, 
                                        device_flag,
                                        6L),
           "double" = cpp_gpuMatrix_pmcc(A@address, 
                                         B@address, 
                                         device_flag,
                                         8L)
    )
    
    return(B)
}

# GPU crossprod
gpu_crossprod <- function(X, Y){
    
    if(nrow(X) != nrow(Y)){
        stop("matrices non-conformable")
    }
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(X)
    
    Z <- gpuMatrix(nrow = ncol(X), ncol = ncol(Y), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = {
               cpp_gpuMatrix_crossprod(X@address, 
                                       Y@address, 
                                       Z@address,
                                       device_flag, 
                                       6L)
               },
           "double" = {
               cpp_gpuMatrix_crossprod(X@address, 
                                       Y@address, 
                                       Z@address, 
                                       device_flag, 8L)
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
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(X)
    
    Z <- gpuMatrix(nrow = nrow(X), ncol = nrow(Y), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = {
               cpp_gpuMatrix_tcrossprod(X@address, 
                                        Y@address, 
                                        Z@address, 
                                        device_flag,
                                        6L)
               },
           "double" = {
               cpp_gpuMatrix_tcrossprod(X@address,
                                        Y@address, 
                                        Z@address, 
                                        device_flag,
                                        8L)
           },
           stop("unsupported type")
    )
    
    return(Z)
}

# GPU Euclidean Distance
gpuMatrix_euclidean <- function(A, D, diag, upper, p, squareDist){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(D)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_gpuMatrix_eucl(A@address, 
                                        D@address, 
                                        squareDist,
                                        device_flag,
                                        6L),
           "double" = cpp_gpuMatrix_eucl(A@address, 
                                         D@address,
                                         squareDist,
                                         device_flag,
                                         8L),
           stop("Unsupported matrix type")
    )
    
    invisible(D)
}


# GPU Pairwise Euclidean Distance
gpuMatrix_peuclidean <- function(A, B, D, squareDist){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(D)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_gpuMatrix_peucl(A@address,
                                         B@address,
                                         D@address, 
                                         squareDist, 
                                         device_flag,
                                         6L),
           "double" = cpp_gpuMatrix_peucl(A@address, 
                                          B@address,
                                          D@address,
                                          squareDist,
                                          device_flag,
                                          8L),
           stop("Unsupported matrix type")
    )
    
    invisible(D)
}

# GPU Element-Wise Absolute Value
gpuMatElemAbs <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuMatrix_elem_abs(A@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuMatrix_elem_abs(A@address,
                                            C@address,
                                            device_flag,
                                            8L)
               }
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
                double = {
                    if(!deviceHasDouble()){
                        stop("Selected GPU does not support double precision")
                    }else{cpp_gpuMatrix_max(A@address, 8L)
                    }
                },
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
                double = {
                    if(!deviceHasDouble()){
                        stop("Selected GPU does not support double precision")
                    }else{cpp_gpuMatrix_min(A@address, 8L)
                    }
                },
                stop("type not recognized")
    )
    return(C)
}

# GPU Matrix transpose
gpuMatrix_t <- function(A){
    
    type <- typeof(A)
    
    B <- gpuMatrix(0, ncol = nrow(A), nrow = ncol(A), type = type)
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    switch(type,
           integer = {cpp_gpuMatrix_transpose(A@address, B@address, device_flag, 4L)},
           float = {cpp_gpuMatrix_transpose(A@address, B@address, device_flag,  6L)},
           double = {cpp_gpuMatrix_transpose(A@address, B@address, device_flag,  8L)},
           stop("type not recognized")
    )
    
    return(B)
}

