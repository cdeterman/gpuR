### gpuVector Wrappers ###

# GPU axpy wrapper
gpuVec_axpy <- function(alpha, A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
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
    
    Z <- gpuVector(length=length(A), type=type)
    if(!missing(B))
    {
        Z@address <- B@address
    }
    
    switch(type,
           integer = {cpp_gpuVector_iaxpy(alpha, 
                                          A@address,
                                          Z@address, 
                                          kernel)
           },
           float = {cpp_vienna_gpuVector_saxpy(alpha, 
                                               A@address, 
                                               Z@address, 
                                               device_flag)
           },
           double = {cpp_vienna_gpuVector_daxpy(alpha, 
                                                A@address,
                                                Z@address,
                                                device_flag)
           },
           stop("type not recognized")
    )
    
    return(Z)
}

# GPU Vector Inner Product
gpuVecInnerProd <- function(A, B){
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    out <- switch(type,
                  "integer" = stop("integer not currently implemented"),
                  "float" = cpp_vienna_gpuVector_sgev_inner(A@address, 
                                                            B@address,
                                                            device_flag),
                  "double" = cpp_vienna_gpuVector_dgev_inner(A@address,
                                                             B@address,
                                                             device_flag),
                  stop("unrecognized data type")
    )
    
    return(as.matrix(out))
}

# GPU Vector Inner Product
gpuVecOuterProd <- function(A, B, C){
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=length(A), ncol=length(B), type=type)
    
    switch(type,
           "integer" = stop("integer not currently implemented"),
           "float" = cpp_vienna_gpuVector_sgev_outer(A@address, 
                                                     B@address,
                                                     C@address,
                                                     device_flag),
           "double" = cpp_vienna_gpuVector_dgev_outer(A@address,
                                                      B@address,
                                                      C@address,
                                                      device_flag),
           stop("unrecognized data type")
    )
    return(C)
}

# GPU Element-Wise Multiplication
gpuVecElemMult <- function(A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
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
           float = {cpp_vienna_sgpuVector_elem_prod(A@address,
                                                    B@address,
                                                    C@address,
                                                    device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuVector_elem_prod(A@address,
                                                     B@address,
                                                     C@address,
                                                     device_flag)
               }
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Division
gpuVecElemDiv <- function(A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
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
           float = {cpp_vienna_sgpuVector_elem_div(A@address,
                                                   B@address,
                                                   C@address,
                                                   device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuVector_elem_div(A@address,
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
gpuVecElemSin <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
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
           float = {cpp_vienna_sgpuVector_elem_sin(A@address,
                                                   C@address,
                                                   device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuVector_elem_sin(A@address,
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
gpuVecElemArcSin <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
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
           float = {cpp_vienna_sgpuVector_elem_asin(A@address,
                                                    C@address,
                                                    device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuVector_elem_asin(A@address,
                                                     C@address,
                                                     device_flag)
               }
           },
           stop("type not recognized")
    )
return(C)
}

# GPU Element-Wise Hyperbolic Sine
gpuVecElemHypSin <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
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
           float = {cpp_vienna_sgpuVector_elem_sinh(A@address,
                                                    C@address,
                                                    device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuVector_elem_sinh(A@address,
                                                     C@address,
                                                     device_flag)
               }
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Sine
gpuVecElemCos <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
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
           float = {cpp_vienna_sgpuVector_elem_cos(A@address,
                                                   C@address,
                                                   device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuVector_elem_cos(A@address,
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
gpuVecElemArcCos <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
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
           float = {cpp_vienna_sgpuVector_elem_acos(A@address,
                                                    C@address,
                                                    device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuVector_elem_acos(A@address,
                                                     C@address,
                                                     device_flag)
               }
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Hyperbolic Sine
gpuVecElemHypCos <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
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
           float = {cpp_vienna_sgpuVector_elem_cosh(A@address,
                                                    C@address,
                                                    device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuVector_elem_cosh(A@address,
                                                     C@address,
                                                     device_flag)
               }
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Sine
gpuVecElemTan <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
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
           float = {cpp_vienna_sgpuVector_elem_tan(A@address,
                                                   C@address,
                                                   device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuVector_elem_tan(A@address,
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
gpuVecElemArcTan <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
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
           float = {cpp_vienna_sgpuVector_elem_atan(A@address,
                                                    C@address,
                                                    device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuVector_elem_atan(A@address,
                                                     C@address,
                                                     device_flag)
               }
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Hyperbolic Sine
gpuVecElemHypTan <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
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
           float = {cpp_vienna_sgpuVector_elem_tanh(A@address,
                                                    C@address,
                                                    device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuVector_elem_tanh(A@address,
                                                     C@address,
                                                     device_flag)
               }
           },
           stop("type not recognized")
    )
    return(C)
}
