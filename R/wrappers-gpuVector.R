### gpuVector Wrappers ###

# GPU axpy wrapper
gpuVec_axpy <- function(alpha, A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    pkg_path <- find.package("gpuR", .libPaths())
    file <- file.path(pkg_path, "CL", "basic_axpy.cl")
    
    if(!file_test("-f", file)){
        stop("kernel file does not exist")
    }
    kernel <- readChar(file, file.info(file)$size)
    
    type <- typeof(A)
    
    Z <- deepcopy(B)
    
    switch(type,
           integer = {cpp_gpuVector_iaxpy(alpha, 
                                          A@address,
                                          Z@address, 
                                          kernel,
                                          device_flag)
           },
           float = {cpp_gpuVector_axpy(alpha, 
                                       A@address, 
                                       Z@address, 
                                       device_flag,
                                       6L)
           },
           double = {cpp_gpuVector_axpy(alpha, 
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
gpuVector_unary_axpy <- function(A){
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option")
        )
    
    type <- typeof(A)
    
    Z <- deepcopy(A)
    
    switch(type,
           integer = {
               cpp_gpuVector_unary_axpy(Z@address, 
                                        device_flag,
                                        4L)
           },
           float = {
               cpp_gpuVector_unary_axpy(Z@address, 
                                        device_flag,
                                        6L)
           },
           double = {
               cpp_gpuVector_unary_axpy(Z@address,
                                        device_flag,
                                        8L)
           },
           stop("type not recognized")
    )
    
    return(Z)
}

# GPU Vector Inner Product
gpuVecInnerProd <- function(A, B){
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    out <- switch(type,
                  "integer" = stop("integer not currently implemented"),
                  "float" = cpp_gpuVector_inner_prod(A@address, 
                                                     B@address,
                                                     device_flag,
                                                     6L),
                  "double" = cpp_gpuVector_inner_prod(A@address,
                                                      B@address,
                                                      device_flag,
                                                      8L),
                  stop("unrecognized data type")
    )
    
    return(as.matrix(out))
}

# GPU Vector Inner Product
gpuVecOuterProd <- function(A, B, C){
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=length(A), ncol=length(B), type=type)
    
    switch(type,
           "integer" = stop("integer not currently implemented"),
           "float" = cpp_gpuVector_outer_prod(A@address, 
                                              B@address,
                                              C@address,
                                              device_flag,
                                              6L),
           "double" = cpp_gpuVector_outer_prod(A@address,
                                               B@address,
                                               C@address,
                                               device_flag,
                                               8L),
           stop("unrecognized data type")
    )
    return(C)
}

# GPU Element-Wise Multiplication
gpuVecElemMult <- function(A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    if(length(A) != length(B)){
        stop("arguments not conformable")
    }
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_prod(A@address,
                                            B@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuVector_elem_prod(A@address,
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

# GPU Scalar Element-Wise Multiplication
gpuVecScalarMult <- function(A, B){
    
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
           float = {cpp_gpuVector_scalar_prod(C@address,
                                              B,
                                              device_flag,
                                              6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuVector_scalar_prod(C@address,
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
gpuVecElemDiv <- function(A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    if(length(A) != length(B)){
        stop("arguments not conformable")
    }
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_div(A@address,
                                           B@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuVector_elem_div(A@address,
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

# GPU Scalar Element-Wise Division
gpuVecScalarDiv <- function(A, B, order){
    
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
           float = {cpp_gpuVector_scalar_div(C@address,
                                             B,
                                             order,
                                             device_flag,
                                             6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuVector_scalar_div(C@address,
                                              B,
                                              order,
                                              device_flag,
                                              8L)
               }
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Power
gpuVecElemPow <- function(A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    if(length(A) != length(B)){
        stop("arguments not conformable")
    }
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_pow(A@address,
                                           B@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuVector_elem_pow(A@address,
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
gpuVecScalarPow <- function(A, B, order){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_scalar_pow(A@address,
                                           B,
                                           C@address,
                                           order,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuVector_scalar_pow(A@address,
                                            B,
                                            C@address,
                                            order,
                                            device_flag,
                                            8L)
               }
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Sine
gpuVecElemSin <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_sin(A@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuVector_elem_sin(A@address,
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
gpuVecElemArcSin <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_asin(A@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuVector_elem_asin(A@address,
                                             C@address,
                                             device_flag,
                                             8L)
               }
           },
           stop("type not recognized")
    )
return(C)
}

# GPU Element-Wise Hyperbolic Sine
gpuVecElemHypSin <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_sinh(A@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuVector_elem_sinh(A@address,
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
gpuVecElemCos <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_cos(A@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuVector_elem_cos(A@address,
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
gpuVecElemArcCos <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_acos(A@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuVector_elem_acos(A@address,
                                             C@address,
                                             device_flag,
                                             8L)
               }
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Hyperbolic Sine
gpuVecElemHypCos <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_cosh(A@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuVector_elem_cosh(A@address,
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
gpuVecElemTan <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_tan(A@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuVector_elem_tan(A@address,
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
gpuVecElemArcTan <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_atan(A@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuVector_elem_atan(A@address,
                                             C@address,
                                             device_flag,
                                             8L)
               }
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Hyperbolic Sine
gpuVecElemHypTan <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_tanh(A@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuVector_elem_tanh(A@address,
                                             C@address,
                                             device_flag,
                                             8L)
               }
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Log10
gpuVecElemLog10 <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_log10(A@address,
                                             C@address,
                                             device_flag,
                                             6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuVector_elem_log10(A@address,
                                              C@address,
                                              device_flag,
                                              8L)
               }
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Natural Log
gpuVecElemLog <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuVector(length = length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_log(A@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuVector_elem_log(A@address,
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
gpuVecElemLogBase <- function(A, base){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuVector(length = length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_log_base(A@address,
                                                C@address,
                                                base,
                                                device_flag,
                                                6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuVector_elem_log_base(A@address,
                                                 C@address,
                                                 base,
                                                 device_flag,
                                                 8L)
               }
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Exponential
gpuVecElemExp <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_exp(A@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuVector_elem_exp(A@address,
                                            C@address,
                                            device_flag,
                                            8L)
               }
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Absolute Value
gpuVecElemAbs <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_abs(A@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_gpuVector_elem_abs(A@address,
                                            C@address,
                                            device_flag,
                                            8L)
               }
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Vector maximum
gpuVecMax <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- switch(type,
                integer = {
                    stop("integer not currently implemented")
                },
                float = {cpp_gpuVector_max(A@address,
                                           device_flag,
                                           6L)
                },
                double = {
                    if(!deviceHasDouble()){
                        stop("Selected GPU does not support double precision")
                    }else{cpp_gpuVector_max(A@address,
                                            device_flag,
                                            8L)
                    }
                },
                stop("type not recognized")
    )
    return(C)
}

# GPU Vector minimum
gpuVecMin <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- switch(type,
                integer = {
                    stop("integer not currently implemented")
                },
                float = {cpp_gpuVector_min(A@address,
                                           device_flag,
                                           6L)
                },
                double = {
                    if(!deviceHasDouble()){
                        stop("Selected GPU does not support double precision")
                    }else{cpp_gpuVector_min(A@address,
                                            device_flag,
                                            8L)
                    }
                },
                stop("type not recognized")
    )
    return(C)
}
