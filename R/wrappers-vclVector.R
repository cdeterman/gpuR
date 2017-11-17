

# GPU Element-Wise Sine
vclVecElemSin <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- vclVector(length=length(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_sin(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               cpp_vclVector_elem_sin(A@address,
                                      C@address,
                                      8L)
           },
           stop("type not recognized")
    )
    
    if(inplace){
        return(invisible(C))
    }else{
        return(C)    
    }
}


# GPU Element-Wise Arc Sine
vclVecElemArcSin <- function(A, inplace = FALSE){
    
    type <- typeof(A)

    if(inplace){
        C <- A
    }else{
        C <- vclVector(length=length(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_asin(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_vclVector_elem_asin(A@address,
                                       C@address,
                                       8L)
           },
           stop("type not recognized")
           )
    
    if(inplace){
        return(invisible(C))
    }else{
        return(C)    
    }
}


# GPU Element-Wise Hyperbolic Sine
vclVecElemHypSin <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- vclVector(length=length(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_sinh(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_vclVector_elem_sinh(A@address,
                                       C@address,
                                       8L)
               
           },
           stop("type not recognized")
           )
    
    if(inplace){
        return(invisible(C))
    }else{
        return(C)
    }
}


# GPU Element-Wise Cos
vclVecElemCos <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- vclVector(length=length(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_cos(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               cpp_vclVector_elem_cos(A@address,
                                      C@address,
                                      8L)
           },
           stop("type not recognized")
           )
    
    if(inplace){
        return(invisible(C))
    }else{
        return(C)    
    }
}


# GPU Element-Wise Arc Cos
vclVecElemArcCos <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- vclVector(length=length(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_acos(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_vclVector_elem_acos(A@address,
                                       C@address,
                                       8L)
           },           
           stop("type not recognized")
    )
    
    if(inplace){
        return(invisible(C))
    }else{
        return(C)    
    }
}


# GPU Element-Wise Hyperbolic Cos
vclVecElemHypCos <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- vclVector(length=length(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_cosh(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_vclVector_elem_cosh(A@address,
                                       C@address,
                                       8L)
           },
           stop("type not recognized")
    )
    
    if(inplace){
        return(invisible(C))
    }else{
        return(C)
    }
}


# GPU Element-Wise Tan
vclVecElemTan <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- vclVector(length=length(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_tan(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_vclVector_elem_tan(A@address,
                                             C@address,
                                             8L)
           },
           stop("type not recognized")
    )
    
    if(inplace){
        return(invisible(C))
    }else{
        return(C)    
    }
}


# GPU Element-Wise Arc Tan
vclVecElemArcTan <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
       C <- A 
    }else{
        C <- vclVector(length=length(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_atan(A@address,
                                             C@address,
                                             6L)
           },
           double = {
               cpp_vclVector_elem_atan(A@address,
                                              C@address,
                                              8L)
           },
           stop("type not recognized")
    )
    
    if(inplace){
        return(invisible(C))
    }else{
        return(C)    
    }
}


# GPU Element-Wise Hyperbolic Tan
vclVecElemHypTan <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- vclVector(length=length(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_tanh(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_vclVector_elem_tanh(A@address,
                                             C@address,
                                             8L)
           },
           stop("type not recognized")
    )
    
    if(inplace){
        return(invisible(C))
    }else{
        return(C)    
    }
}


# GPU Element-Wise Natural Log
vclVecElemLog <- function(A){
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_log(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               cpp_vclVector_elem_log(A@address,
                                            C@address,
                                            8L)
           },
           stop("type not recognized")
    )
    return(C)
}


# GPU Element-Wise Log Base
vclVecElemLogBase <- function(A, base){
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_log_base(A@address,
                                                C@address,
                                                base,
                                                6L)
           },
           double = {
               cpp_vclVector_elem_log_base(A@address,
                                                 C@address,
                                                 base,
                                                 8L)
           },
           stop("type not recognized")
           )
    return(C)
}


# GPU Element-Wise Base 10 Log
vclVecElemLog10 <- function(A){
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_log10(A@address,
                                             C@address,
                                             6L)
           },
           double = {
               cpp_vclVector_elem_log10(A@address,
                                              C@address,
                                              8L)
           },
           stop("type not recognized")
    )
    return(C)
}


# GPU Element-Wise Exponential
vclVecElemExp <- function(A){
    
    type <- typeof(A)
    
    C <- vclVector(length=length(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_exp(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_vclVector_elem_exp(A@address,
                                             C@address,
                                             8L)
           },
           stop("type not recognized")
    )
    return(C)
}


# GPU Element-Wise Absolute Value
vclVecElemAbs <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- vclVector(length=length(A), type=type, ctx_id = A@.context_index)   
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclVector_elem_abs(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               cpp_vclVector_elem_abs(A@address,
                                            C@address,
                                            8L)
           },
           stop("type not recognized")
    )
    return(C)
}

# vclVecElemAbs2 <- function(A){
# 
#     type <- typeof(A)
# 
#     switch(type,
#            integer = {
#                stop("integer not currently implemented")
#            },
#            float = {
#                file <- system.file("CL", "fabs.cl", package = "gpuR")
# 
#                if(!file_test("-f", file)){
#                    stop("kernel file does not exist")
#                }
#                kernel <- readChar(file, file.info(file)$size)
# 
#                cpp_vclVector_elem_abs2(A@address,
#                                            kernel,
#                                             A@.context_index - 1,
#                                            6L)
#            },
#            double = {
#                file <- system.file("CL", "dabs.cl", package = "gpuR")
# 
#                if(!file_test("-f", file)){
#                    stop("kernel file does not exist")
#                }
#                kernel <- readChar(file, file.info(file)$size)
# 
#                cpp_vclVector_elem_abs2(A@address,
#                                        kernel,
#                                        A@.context_index - 1,
#                                       8L)
#            },
#            stop("type not recognized")
#     )
#     return(invisible(A))
# }

# GPU Element-Wise Absolute Value
vclVecElemMaxAbs <- function(A){
    
    type <- typeof(A)
    
    out <- switch(type,
                  integer = {
                      stop("integer not currently implemented")
                  },
                  float = {cpp_vclVector_elem_max_abs(A@address,
                                                      6L)
                  },
                  double = {
                      cpp_vclVector_elem_max_abs(A@address,
                                                 8L)
                  },
                  stop("type not recognized")
    )
    return(out)
}

# GPU Vector maximum
vclVecMax <- function(A){
    
    type <- typeof(A)
    
    C <- switch(type,
                integer = {
                    stop("integer not currently implemented")
                },
                float = {cpp_vclVector_max(A@address, 6L)
                },
                double = {
                    cpp_vclVector_max(A@address, 8L)
                },
                stop("type not recognized")
    )
    return(C)
}


# GPU Vector minimum
vclVecMin <- function(A){
    
    type <- typeof(A)
    
    C <- switch(type,
                integer = {
                    stop("integer not currently implemented")
                },
                float = {cpp_vclVector_min(A@address, 6L)
                },
                double = {
                    cpp_vclVector_min(A@address, 8L)
                },
                stop("type not recognized")
    )
    return(C)
}


