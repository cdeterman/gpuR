
# vclMatrix GEMM
vclMatMult <- function(A, B){
    
#     pkg_path <- find.package("gpuR", .libPaths())
#     file <- file.path(pkg_path, "CL", "basic_gemm.cl")
#     
#     if(!file_test("-f", file)){
#         stop("kernel file does not exist")
#     }
#     kernel <- readChar(file, file.info(file)$size)
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(B), type=type)
    
    switch(type,
           integer = {
               stop("OpenCL integer GEMM not currently
                    supported for viennacl matrices")
                #cpp_vclMatrix_igemm(A@address,
                #                   B@address, 
                #                   C@address,
                #                   kernel)
               #                      cpp_vclMatrix_igemm(A@address,
               #                                                        B@address,
               #                                                        C@address)
           },
           float = {cpp_vclMatrix_sgemm(A@address,
                                        B@address,
                                        C@address,
                                        device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_dgemm(A@address,
                                         B@address,
                                         C@address,
                                         device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# vclMatrix AXPY
vclMat_axpy <- function(alpha, A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    nrA = nrow(A)
    ncA = ncol(A)
    nrB = nrow(B)
    ncB = ncol(B)
    
#     pkg_path <- find.package("gpuR", .libPaths())
#     file <- file.path(pkg_path, "CL", "basic_axpy.cl")
#     
#     if(!file_test("-f", file)){
#         stop("kernel file does not exist")
#     }
#     kernel <- readChar(file, file.info(file)$size)
    
    type <- typeof(A)
    
    Z <- vclMatrix(nrow=nrB, ncol=ncA, type=type)
    if(!missing(B))
    {
        if(length(B[]) != length(A[])) stop("Lengths of matrices must match")
        Z@address <- B@address
    }
    
    switch(type,
           integer = {
               stop("OpenCL integer GEMM not currently
                    supported for viennacl matrices")
               #cpp_vclMatrix_iaxpy(alpha, 
                #                          A@address,
                #                          Z@address, 
                #                          kernel)
           },
           float = {cpp_vclMatrix_saxpy(alpha, 
                                        A@address, 
                                        Z@address,
                                        device_flag)
           },
           double = {cpp_vclMatrix_daxpy(alpha, 
                                         A@address,
                                         Z@address,
                                         device_flag)
           },
            stop("type not recognized")
    )

return(Z)
}

# vclMatrix crossprod
vcl_crossprod <- function(X, Y){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    if(ncol(X) != ncol(Y)){
        stop("matrices non-conformable")
    }
    
    type <- typeof(X)
    
    Z <- vclMatrix(nrow = ncol(X), ncol = ncol(Y), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vclMatrix_scrossprod(X@address, 
                                              Y@address, 
                                              Z@address,
                                              device_flag),
           "double" = cpp_vclMatrix_dcrossprod(X@address, 
                                               Y@address, 
                                               Z@address,
                                               device_flag)
    )
    
    return(Z)
}

# vclMatrix crossprod
vcl_tcrossprod <- function(X, Y){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    if(nrow(X) != nrow(Y)){
        stop("matrices non-conformable")
    }
    
    type <- typeof(X)
    
    Z <- vclMatrix(nrow = nrow(X), ncol = nrow(Y), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vclMatrix_stcrossprod(X@address,
                                               Y@address, 
                                               Z@address,
                                               device_flag),
           "double" = cpp_vclMatrix_dtcrossprod(X@address, 
                                                Y@address, 
                                                Z@address,
                                                device_flag),
           stop("type not recognized")
    )
    
    return(Z)
}


# GPU Element-Wise Multiplication
vclMatElemMult <- function(A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    if(!all(dim(A) == dim(B))){
        stop("matrices not conformable")
    }
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_svclMatrix_elem_prod(A@address,
                                                    B@address,
                                                    C@address,
                                                    device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_dvclMatrix_elem_prod(A@address,
                                                     B@address,
                                                     C@address,
                                                     device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Division
vclMatElemDiv <- function(A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    if(!all(dim(A) == dim(B))){
        stop("matrices not conformable")
    }
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_svclMatrix_elem_div(A@address,
                                                   B@address,
                                                   C@address,
                                                   device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_dvclMatrix_elem_div(A@address,
                                                    B@address,
                                                    C@address,
                                                    device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}


# GPU Element-Wise Sine
vclMatElemSin <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_svclMatrix_elem_sin(A@address,
                                                   C@address,
                                                   device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_dvclMatrix_elem_sin(A@address,
                                                    C@address,
                                                    device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Arc Sine
vclMatElemArcSin <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_svclMatrix_elem_asin(A@address,
                                                    C@address,
                                                    device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_dvclMatrix_elem_asin(A@address,
                                                     C@address,
                                                     device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Hyperbolic Sine
vclMatElemHypSin <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_svclMatrix_elem_sinh(A@address,
                                                    C@address,
                                                    device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_dvclMatrix_elem_sinh(A@address,
                                                     C@address,
                                                     device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Cos
vclMatElemCos <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_svclMatrix_elem_cos(A@address,
                                                   C@address,
                                                   device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_dvclMatrix_elem_cos(A@address,
                                                    C@address,
                                                    device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Arc Cos
vclMatElemArcCos <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_svclMatrix_elem_acos(A@address,
                                                    C@address,
                                                    device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_dvclMatrix_elem_acos(A@address,
                                                     C@address,
                                                     device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Hyperbolic Cos
vclMatElemHypCos <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_svclMatrix_elem_cosh(A@address,
                                                    C@address,
                                                    device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_dvclMatrix_elem_cosh(A@address,
                                                     C@address,
                                                     device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Tan
vclMatElemTan <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_svclMatrix_elem_tan(A@address,
                                                   C@address,
                                                   device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_dvclMatrix_elem_tan(A@address,
                                                    C@address,
                                                    device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Arc Tan
vclMatElemArcTan <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_svclMatrix_elem_atan(A@address,
                                                    C@address,
                                                    device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_dvclMatrix_elem_atan(A@address,
                                                     C@address,
                                                     device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Hyperbolic Tan
vclMatElemHypTan <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_svclMatrix_elem_tanh(A@address,
                                                    C@address,
                                                    device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_dvclMatrix_elem_tanh(A@address,
                                                     C@address,
                                                     device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Natural Log
vclMatElemLog <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_svclMatrix_elem_log(A@address,
                                                   C@address,
                                                   device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_dvclMatrix_elem_log(A@address,
                                                    C@address,
                                                    device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Log Base
vclMatElemLogBase <- function(A, base){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_svclMatrix_elem_log_base(A@address,
                                                        C@address,
                                                        base,
                                                        device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_dvclMatrix_elem_log_base(A@address,
                                                         C@address,
                                                         base,
                                                         device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Base 10 Log
vclMatElemLog10 <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_svclMatrix_elem_log10(A@address,
                                                     C@address,
                                                     device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_dvclMatrix_elem_log10(A@address,
                                                      C@address,
                                                      device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Exponential
vclMatElemExp <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_svclMatrix_elem_exp(A@address,
                                                   C@address,
                                                   device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_dvclMatrix_elem_exp(A@address,
                                                    C@address,
                                                    device_flag)
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
        switch(options("gpuR.default.device")$gpuR.default.device,
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
           "float" = cpp_fvclMatrix_colsum(A@address, sums@address, device_flag),
           "double" = cpp_dvclMatrix_colsum(A@address, sums@address, device_flag)
    )
    
    return(sums)
}

# GPU rowSums
gpu_rowSums <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
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
           "float" = cpp_fvclMatrix_rowsum(A@address, sums@address, device_flag),
           "double" = cpp_dvclMatrix_rowsum(A@address, sums@address, device_flag)
    )
    
    return(sums)
}

# GPU colMeans
gpu_colMeans <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
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
           "float" = cpp_fvclMatrix_colmean(A@address, sums@address, device_flag),
           "double" = cpp_dvclMatrix_colmean(A@address, sums@address, device_flag)
    )
    
    return(sums)
}

# GPU rowMeans
gpu_rowMeans <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
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
           "float" = cpp_fvclMatrix_rowmean(A@address, sums@address, device_flag),
           "double" = cpp_dvclMatrix_rowmean(A@address, sums@address, device_flag)
    )
    
    return(sums)
}

# GPU Pearson Covariance
gpu_pmcc <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    B <- vclMatrix(nrow = ncol(A), ncol = ncol(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_fvclMatrix_pmcc(A@address, B@address, device_flag),
           "double" = cpp_dvclMatrix_pmcc(A@address, B@address, device_flag)
    )
    
    return(B)
}
