### gpuVector Wrappers ###

# GPU axpy wrapper
gpuVec_axpy <- function(alpha, A, B, inplace = FALSE, order = 0){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    type <- typeof(A)
    
    if(inplace){
        Z <- B
    }else{
        Z <- deepcopy(B)   
    }
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuVector_axpy(alpha,
                                  A@address,
                                  is(A, "vclVector"),
                                  Z@address,
                                  is(Z, "vclVector"),
                                  order,
                                  4L,
                                  A@.context_index - 1)
           },
           float = {cpp_gpuVector_axpy(alpha, 
                                       A@address, 
                                       is(A, "vclVector"),
                                       Z@address,
                                       is(Z, "vclVector"),
                                       order,
                                       6L,
                                       A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_axpy(alpha, 
                                  A@address,
                                  is(A, "vclVector"),
                                  Z@address,
                                  is(Z, "vclVector"),
                                  order,
                                  8L,
                                  A@.context_index - 1)
           },
           stop("type not recognized")
    )
    
    if(inplace){
        return(invisible(Z))
    }else{
        return(Z)    
    }
}

# GPU axpy wrapper
gpuVector_unary_axpy <- function(A){
    
    type <- typeof(A)
    
    Z <- deepcopy(A)
    
    switch(type,
           integer = {
               cpp_gpuVector_unary_axpy(Z@address, 
                                        is(Z, "vclVector"),
                                        4L)
           },
           float = {
               cpp_gpuVector_unary_axpy(Z@address, 
                                        is(Z, "vclVector"),
                                        6L,
                                        A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_unary_axpy(Z@address,
                                        is(Z, "vclVector"),
                                        8L,
                                        A@.context_index - 1)
           },
           stop("type not recognized")
    )
    
    return(Z)
}

# GPU Vector Inner Product
gpuVecInnerProd <- function(A, B){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    type <- typeof(A)
    
    out <- switch(type,
                  "integer" = stop("integer not currently implemented"),
                  "float" = cpp_gpuVector_inner_prod(A@address, 
                                                     is(A, "vclVector"),
                                                     B@address,
                                                     is(B, "vclVector"),
                                                     6L,
                                                     A@.context_index - 1),
                  "double" = {
                      cpp_gpuVector_inner_prod(A@address,
                                               is(A, "vclVector"),
                                               B@address,
                                               is(B, "vclVector"),
                                               8L,
                                               A@.context_index - 1)
                  },
                  stop("unrecognized data type")
    )
    
    return(as.matrix(out))
}

# GPU Vector Inner Product
gpuVecOuterProd <- function(A, B, C){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    type <- typeof(A)
    
    if(is(A, "vclVector")){
        C <- vclMatrix(nrow=length(A), ncol=length(B), type=type, ctx_id = A@.context_index)
    }else{
        C <- gpuMatrix(nrow=length(A), ncol=length(B), type=type, ctx_id = A@.context_index)
    }
    
    
    switch(type,
           "integer" = stop("integer not currently implemented"),
           "float" = cpp_gpuVector_outer_prod(A@address, 
                                              is(A, "vclVector"),
                                              B@address,
                                              is(B, "vclVector"),
                                              C@address,
                                              is(C, "vclMatrix"),
                                              6L,
                                              A@.context_index - 1),
           "double" = {
               cpp_gpuVector_outer_prod(A@address,
                                        is(A, "vclVector"),
                                        B@address,
                                        is(B, "vclVector"),
                                        C@address,
                                        is(C, "vclMatrix"),
                                        8L,
                                        A@.context_index - 1)
           },
           stop("unrecognized data type")
    )
    return(C)
}

# GPU Element-Wise Multiplication
gpuVecElemMult <- function(A, B, inplace = FALSE){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    if(length(A) != length(B)){
        stop("arguments not conformable")
    }
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        if(is(A, "vclVector")){
            C <- vclVector(length=length(A), type=type, ctx_id = A@.context_index)   
        }else{
            C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)    
        }
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_prod(A@address,
                                            is(A, "vclVector"),
                                            B@address,
                                            is(B, "vclVector"),
                                            C@address,
                                            is(C, "vclVector"),
                                            6L,
                                            A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_elem_prod(A@address,
                                       is(A, "vclVector"),
                                       B@address,
                                       is(B, "vclVector"),
                                       C@address,
                                       is(C, "vclVector"),
                                       8L,
                                       A@.context_index - 1)
           },
           stop("type not recognized")
    )
    
    if(inplace){
        return(invisible(C))
    }else{
        return(C)    
    }
}

# GPU Scalar Element-Wise Multiplication
gpuVecScalarMult <- function(A, B, inplace = FALSE){
    
    # quick class check when scalars are passed
    if(inherits(A, "gpuVector") | inherits(A, "vclVector")){
        
        type <- typeof(A)
        
        if(inplace){
            C <- A
        }else{
            C <- deepcopy(A)
        }    
        Z <- B
    }else{
        
        type <- typeof(B)
        
        if(inplace){
            C <- B
        }else{
            C <- deepcopy(B)
        }
        Z <- A
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {
               cpp_gpuVector_scalar_prod(C@address,
                                         is(C, "vclVector"),
                                         Z,
                                         6L,
                                         C@.context_index - 1)
           },
           double = {
               cpp_gpuVector_scalar_prod(C@address,
                                         is(C, "vclVector"),
                                         Z,
                                         8L,
                                         C@.context_index - 1)
           },
           stop("type not recognized")
    )
    
    if(inplace){
        return(invisible(C))
    }else{
        return(C)    
    }
}

# GPU Element-Wise Division
gpuVecElemDiv <- function(A, B, inplace = FALSE){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    if(length(A) != length(B)){
        stop("arguments not conformable")
    }
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        if(is(A, "vclVector")){
            C <- vclVector(length=length(A), type=type, ctx_id = A@.context_index)   
        }else{
            C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)    
        }
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_div(A@address,
                                           is(A, "vclVector"),
                                           B@address,
                                           is(B, "vclVector"),
                                           C@address,
                                           is(C, "vclVector"),
                                           6L,
                                           A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_elem_div(A@address,
                                      is(A, "vclVector"),
                                      B@address,
                                      is(B, "vclVector"),
                                      C@address,
                                      is(C, "vclVector"),
                                      8L,
                                      A@.context_index - 1)
           },
           
           stop("type not recognized")
    )
    return(C)
}

# GPU Scalar Element-Wise Division
gpuVecScalarDiv <- function(A, B, order=0, inplace = FALSE){
    
    # quick class check when scalars are passed
    if(inherits(A, "gpuVector") | inherits(A, "vclVector")){
        
        type <- typeof(A)
        
        if(inplace){
            C <- A
        }else{
            C <- deepcopy(A)
        }    
        Z <- B
    }else{
        
        type <- typeof(B)
        
        if(inplace){
            C <- B
        }else{
            C <- deepcopy(B)
        }
        Z <- A
    }
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuVector_scalar_div(C@address,
                                        is(C, "vclVector"),
                                        Z,
                                        order,
                                        4L,
                                        C@.context_index - 1)
           },
           float = {cpp_gpuVector_scalar_div(C@address,
                                             is(C, "vclVector"),
                                             Z,
                                             order,
                                             6L,
                                             C@.context_index - 1)
           },
           double = {
               cpp_gpuVector_scalar_div(C@address,
                                        is(C, "vclVector"),
                                        Z,
                                        order,
                                        8L,
                                        C@.context_index - 1)
           },
           stop("type not recognized")
    )
    
    if(inplace){
        return(invisible(C))
    }else{
        return(C)    
    }
}

# GPU Element-Wise Power
gpuVecElemPow <- function(A, B){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    if(length(A) != length(B)){
        stop("arguments not conformable")
    }
    
    type <- typeof(A)
    
    if(is(A, "vclVector")){
        C <- vclVector(length=length(A), type=type, ctx_id = A@.context_index)   
    }else{
        C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)    
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_pow(A@address,
                                           is(A, "vclVector"),
                                           B@address,
                                           is(B, "vclVector"),
                                           C@address,
                                           is(C, "vclVector"),
                                           6L,
                                           A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_elem_pow(A@address,
                                      is(A, "vclVector"),
                                      B@address,
                                      is(B, "vclVector"),
                                      C@address,
                                      is(C, "vclVector"),
                                      8L,
                                      A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Power
gpuVecScalarPow <- function(A, B, order){
    
    type <- typeof(A)
    
    if(is(A, "vclVector")){
        C <- vclVector(length=length(A), type=type, ctx_id = A@.context_index)   
    }else{
        C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)    
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_scalar_pow(A@address,
                                             is(A, "vclVector"),
                                             B,
                                             C@address,
                                             is(C, "vclVector"),
                                             order,
                                             6L,
                                             A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_scalar_pow(A@address,
                                        is(A, "vclVector"),
                                        B,
                                        C@address,
                                        is(C, "vclVector"),
                                        order,
                                        8L,
                                        A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise sqrt
gpuVecSqrt <- function(A){
    
    type <- typeof(A)
    
    if(is(A, "vclVector")){
        C <- vclVector(length=length(A), type=type, ctx_id = A@.context_index)   
    }else{
        C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)    
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_sqrt(A@address,
                                       is(A, "vclVector"),
                                       C@address,
                                       is(C, "vclVector"),
                                       6L,
                                       A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_sqrt(A@address,
                                  is(A, "vclVector"),
                                  C@address,
                                  is(C, "vclVector"),
                                  8L,
                                  A@.context_index - 1)
           },
           {
               stop("type not recognized")
           })
    return(C)
}

# GPU Element-Wise Sine
gpuVecElemSin <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_sin(A@address,
                                           C@address,
                                           6L,
                                           A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_elem_sin(A@address,
                                      C@address,
                                      8L,
                                      A@.context_index - 1)
           },
           {
               stop("type not recognized")
           })
    
    if(inplace){
        return(invisible(C))
    }else{
        return(C)
    }
}

# GPU Element-Wise Arc Sine
gpuVecElemArcSin <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_asin(A@address,
                                            C@address,
                                            6L,
                                            A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_elem_asin(A@address,
                                       C@address,
                                       8L,
                                       A@.context_index - 1)
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
gpuVecElemHypSin <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_sinh(A@address,
                                            C@address,
                                            6L,
                                            A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_elem_sinh(A@address,
                                       C@address,
                                       8L,
                                       A@.context_index - 1)
           },
           stop("type not recognized")
    )
    
    if(inplace){
        return(invisible(C))
    }else{
        return(C)   
    }
}

# GPU Element-Wise Sine
gpuVecElemCos <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_cos(A@address,
                                           C@address,
                                           6L,
                                           A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_elem_cos(A@address,
                                      C@address,
                                      8L,
                                      A@.context_index - 1)
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
gpuVecElemArcCos <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_acos(A@address,
                                            C@address,
                                            6L,
                                            A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_elem_acos(A@address,
                                       C@address,
                                       8L,
                                       A@.context_index - 1)
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
gpuVecElemHypCos <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_cosh(A@address,
                                            C@address,
                                            6L,
                                            A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_elem_cosh(A@address,
                                       C@address,
                                       8L,
                                       A@.context_index - 1)
           },
           stop("type not recognized")
    )
    
    if(inplace){
        return(invisible(C))
    }else{
        return(C)    
    }
}

# GPU Element-Wise Sine
gpuVecElemTan <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_tan(A@address,
                                           C@address,
                                           6L,
                                           A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_elem_tan(A@address,
                                      C@address,
                                      8L,
                                      A@.context_index - 1)
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
gpuVecElemArcTan <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_atan(A@address,
                                            C@address,
                                            6L,
                                            A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_elem_atan(A@address,
                                       C@address,
                                       8L,
                                       A@.context_index - 1)
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
gpuVecElemHypTan <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_tanh(A@address,
                                            C@address,
                                            6L,
                                            A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_elem_tanh(A@address,
                                       C@address,
                                       8L,
                                       A@.context_index - 1)
           },
           stop("type not recognized")
    )
    
    if(inplace){
        return(invisible(C))
    }else{
        return(C)    
    }
}

# GPU Element-Wise Log10
gpuVecElemLog10 <- function(A){
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_log10(A@address,
                                             C@address,
                                             6L,
                                             A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_elem_log10(A@address,
                                        C@address,
                                        8L,
                                        A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Natural Log
gpuVecElemLog <- function(A){
    
    type <- typeof(A)
    
    C <- gpuVector(length = length(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_log(A@address,
                                           C@address,
                                           6L,
                                           A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_elem_log(A@address,
                                      C@address,
                                      8L,
                                      A@.context_index - 1)
           },
           
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Log Base
gpuVecElemLogBase <- function(A, base){
    
    type <- typeof(A)
    
    C <- gpuVector(length = length(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_log_base(A@address,
                                                C@address,
                                                base,
                                                6L,
                                                A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_elem_log_base(A@address,
                                           C@address,
                                           base,
                                           8L,
                                           A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Exponential
gpuVecElemExp <- function(A){
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_exp(A@address,
                                           C@address,
                                           6L,
                                           A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_elem_exp(A@address,
                                      C@address,
                                      8L,
                                      A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Absolute Value
gpuVecElemAbs <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_abs(A@address,
                                           C@address,
                                           6L,
                                           A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_elem_abs(A@address,
                                      C@address,
                                      8L,
                                      A@.context_index - 1)
           },
           stop("type not recognized")
    )
    
    if(inplace){
        return(invisible(C))
    }else{
        return(C)    
    }
}

# GPU Vector maximum
gpuVecMax <- function(A){
    
    type <- typeof(A)
    
    C <- switch(type,
                integer = {
                    stop("integer not currently implemented")
                },
                float = {cpp_gpuVector_max(A@address,
                                           6L,
                                           A@.context_index - 1)
                },
                double = {
                    cpp_gpuVector_max(A@address,
                                      8L,
                                      A@.context_index - 1)
                },
                stop("type not recognized")
    )
    return(C)
}

# GPU Vector minimum
gpuVecMin <- function(A){
    
    type <- typeof(A)
    
    C <- switch(type,
                integer = {
                    stop("integer not currently implemented")
                },
                float = {cpp_gpuVector_min(A@address,
                                           6L,
                                           A@.context_index - 1)
                },
                double = {
                    cpp_gpuVector_min(A@address,
                                      8L,
                                      A@.context_index - 1)
                },
                stop("type not recognized")
    )
    return(C)
}
