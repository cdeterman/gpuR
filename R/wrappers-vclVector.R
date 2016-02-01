
# vclVector Inner (Dot) Product
vclVecInner <- function(A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    out <- switch(type,
                  integer = {
                      stop("OpenCL integer dot product not currently
                    supported for viennacl matrices")
                      #cpp_vclVector_igemm(A@address,
                      #                   B@address, 
                      #                   C@address,
                      #                   kernel)
                      #                      cpp_vclVector_igemm(A@address,
                      #                                                        B@address,
                      #                                                        C@address)
                  },
                  float = {cpp_vclVector_inner_prod(A@address,
                                                    B@address,
                                                    device_flag,
                                                    6L)
                  },
                  double = {
                      if(!deviceHasDouble()){
                          stop("Selected GPU does not support double precision")
                      }else{cpp_vclVector_inner_prod(A@address,
                                                     B@address,
                                                     device_flag,
                                                     8L)
                      }
                  },
                  stop("type not recognized")
                  )
    return(as.matrix(out))
}


# vclVector Outer Product
vclVecOuter <- function(A, B){
    
    if(length(B) != length(A)) stop("Non conformant arguments")
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=length(A), ncol=length(B), type=type)
    
    switch(type,
           integer = {
               stop("OpenCL integer outer product not currently
                    supported for viennacl matrices")
               #cpp_vclVector_igemm(A@address,
               #                   B@address, 
               #                   C@address,
               #                   kernel)
               #                      cpp_vclVector_igemm(A@address,
               #                                                        B@address,
               #                                                        C@address)
           },
           float = {cpp_vclVector_outer_prod(A@address,
                                             B@address,
                                             C@address,
                                             device_flag,
                                             6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_outer_prod(A@address,
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

# vclVector AXPY
vclVec_axpy <- function(alpha, A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    Z <- vclVector(length=length(A), type=type)
    if(!missing(B))
    {
        if(length(B) != length(A)) stop("Lengths of matrices must match")
        Z <- deepcopy(B)
    }
    
    switch(type,
           integer = {
               stop("OpenCL integer AXPY not currently
                    supported for viennacl matrices")
               #cpp_vclVector_iaxpy(alpha, 
               #                          A@address,
               #                          Z@address, 
               #                          kernel)
           },
           float = {cpp_vclVector_axpy(alpha, 
                                       A@address, 
                                       Z@address,
                                       device_flag,
                                       6L)
           },
           double = {cpp_vclVector_axpy(alpha, 
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
vclVector_unary_axpy <- function(A){
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
               cpp_vclVector_unary_axpy(Z@address, 
                                        device_flag,
                                        4L)
           },
           float = {
               cpp_vclVector_unary_axpy(Z@address, 
                                        device_flag,
                                        6L)
           },
           double = {
               cpp_vclVector_unary_axpy(Z@address,
                                        device_flag,
                                        8L)
           },
           stop("type not recognized")
    )
    
    return(Z)
}

# GPU Element-Wise Multiplication
vclVecElemMult <- function(A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    if( length(A) != length(B)){
        stop("Non-conformant arguments")
    }
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_prod(A@address,
                                            B@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_elem_prod(A@address,
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
vclVecScalarMult <- function(A, B){
    
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
           float = {cpp_vclVector_scalar_prod(C@address,
                                              B,
                                              device_flag,
                                              6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_scalar_prod(C@address,
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
vclVecElemDiv <- function(A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    if( length(A) != length(B)){
        stop("Non-conformant arguments")
    }
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_div(A@address,
                                           B@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_elem_div(A@address,
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
vclVecScalarDiv <- function(A, B){
    
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
           float = {cpp_vclVector_scalar_div(C@address,
                                             B,
                                             device_flag,
                                             6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_scalar_div(C@address,
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
vclVecElemPow <- function(A, B){
    
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
    
    C <- vclVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_pow(A@address,
                                           B@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_elem_pow(A@address,
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
vclVecScalarPow <- function(A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_scalar_pow(A@address,
                                             B,
                                             C@address,
                                             device_flag,
                                             6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_scalar_pow(A@address,
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
vclVecElemSin <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_sin(A@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_elem_sin(A@address,
                                            C@address,
                                            device_flag,
                                            8L)
               }
           },
           stop("type not recognized")
           )
return(C)
}

# GPU Element-Wise Arc Sine
vclVecElemArcSin <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_asin(A@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_elem_asin(A@address,
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
vclVecElemHypSin <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_sinh(A@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_elem_sinh(A@address,
                                             C@address,
                                             device_flag,
                                             8L)
               }
           },
           stop("type not recognized")
           )
    return(C)
}

# GPU Element-Wise Cos
vclVecElemCos <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_cos(A@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_elem_cos(A@address,
                                            C@address,
                                            device_flag,
                                            8L)
               }
           },
           stop("type not recognized")
           )
    return(C)
}

# GPU Element-Wise Arc Cos
vclVecElemArcCos <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_acos(A@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_elem_acos(A@address,
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
vclVecElemHypCos <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_cosh(A@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_elem_cosh(A@address,
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
vclVecElemTan <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_tan(A@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_elem_tan(A@address,
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
vclVecElemArcTan <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_atan(A@address,
                                             C@address,
                                             device_flag,
                                             6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_elem_atan(A@address,
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
vclVecElemHypTan <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_tanh(A@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_elem_tanh(A@address,
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
vclVecElemLog <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_log(A@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_elem_log(A@address,
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

# GPU Element-Wise Log Base
vclVecElemLogBase <- function(A, base){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_log_base(A@address,
                                                C@address,
                                                base,
                                                device_flag,
                                                6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_elem_log_base(A@address,
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

# GPU Element-Wise Base 10 Log
vclVecElemLog10 <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_log10(A@address,
                                             C@address,
                                             device_flag,
                                             6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_elem_log10(A@address,
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
vclVecElemExp <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_exp(A@address,
                                            C@address,
                                            device_flag,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_elem_exp(A@address,
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
vclVecElemAbs <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device.type")$gpuR.default.device.type,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_abs(A@address,
                                           C@address,
                                           device_flag,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclVector_elem_abs(A@address,
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
vclVecMax <- function(A){
    
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
                float = {cpp_vclVector_max(A@address,
                                           device_flag,
                                           6L)
                },
                double = {
                    if(!deviceHasDouble()){
                        stop("Selected GPU does not support double precision")
                    }else{cpp_vclVector_max(A@address,
                                            device_flag,
                                            8L)
                    }
                },
                stop("type not recognized")
    )
    return(C)
}

# GPU Vector minimum
vclVecMin <- function(A){
    
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
                float = {cpp_vclVector_min(A@address,
                                           device_flag,
                                           6L)
                },
                double = {
                    if(!deviceHasDouble()){
                        stop("Selected GPU does not support double precision")
                    }else{cpp_vclVector_min(A@address,
                                            device_flag,
                                            8L)
                    }
                },
                stop("type not recognized")
    )
    return(C)
}


