### gpuVector Wrappers ###

# GPU axpy wrapper
gpuVec_axpy <- function(alpha, A, B){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    type <- typeof(A)
    
    Z <- deepcopy(B)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuVector_axpy(alpha,
                                   A@address,
                                   Z@address,
                                   4L,
                                   A@.context_index - 1)
           },
           float = {cpp_gpuVector_axpy(alpha, 
                                       A@address, 
                                       Z@address, 
                                       6L,
                                       A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_axpy(alpha, 
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
gpuVector_unary_axpy <- function(A){
    
    type <- typeof(A)
    
    Z <- deepcopy(A)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
               cpp_gpuVector_unary_axpy(Z@address, 
                                        4L)
           },
           float = {
               cpp_gpuVector_unary_axpy(Z@address, 
                                        6L,
                                        A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_unary_axpy(Z@address,
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
                                                     B@address,
                                                     6L,
                                                     A@.context_index - 1),
                  "double" = {
                      cpp_gpuVector_inner_prod(A@address,
                                               B@address,
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
    
    C <- gpuMatrix(nrow=length(A), ncol=length(B), type=type, ctx_id = A@.context_index)
    
    switch(type,
           "integer" = stop("integer not currently implemented"),
           "float" = cpp_gpuVector_outer_prod(A@address, 
                                              B@address,
                                              C@address,
                                              6L,
                                              A@.context_index - 1),
           "double" = {
               cpp_gpuVector_outer_prod(A@address,
                                        B@address,
                                        C@address,
                                        8L,
                                        A@.context_index - 1)
           },
           stop("unrecognized data type")
    )
    return(C)
}

# GPU Element-Wise Multiplication
gpuVecElemMult <- function(A, B){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    if(length(A) != length(B)){
        stop("arguments not conformable")
    }
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_prod(A@address,
                                            B@address,
                                            C@address,
                                            6L,
                                            A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_elem_prod(A@address,
                                       B@address,
                                       C@address,
                                       8L,
                                       A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Scalar Element-Wise Multiplication
gpuVecScalarMult <- function(A, B){
    
    type <- typeof(A)
    
    C <- deepcopy(A)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_scalar_prod(C@address,
                                              B,
                                              6L,
                                              A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_scalar_prod(C@address,
                                         B,
                                         8L,
                                         A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Division
gpuVecElemDiv <- function(A, B){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    if(length(A) != length(B)){
        stop("arguments not conformable")
    }
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_div(A@address,
                                           B@address,
                                           C@address,
                                           6L,
                                           A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_elem_div(A@address,
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
gpuVecScalarDiv <- function(A, B, order){
    
    type <- typeof(A)
    
    C <- deepcopy(A)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_scalar_div(C@address,
                                             B,
                                             order,
                                             6L,
                                             A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_scalar_div(C@address,
                                        B,
                                        order,
                                        8L,
                                        A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Power
gpuVecElemPow <- function(A, B){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    if(length(A) != length(B)){
        stop("arguments not conformable")
    }
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_elem_pow(A@address,
                                           B@address,
                                           C@address,
                                           6L,
                                           A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_elem_pow(A@address,
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
gpuVecScalarPow <- function(A, B, order){
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_gpuVector_scalar_pow(A@address,
                                             B,
                                             C@address,
                                             order,
                                             6L,
                                             A@.context_index - 1)
           },
           double = {
               cpp_gpuVector_scalar_pow(A@address,
                                        B,
                                        C@address,
                                        order,
                                        8L,
                                        A@.context_index - 1)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Sine
gpuVecElemSin <- function(A){
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    
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
    return(C)
}

# GPU Element-Wise Arc Sine
gpuVecElemArcSin <- function(A){
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    
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
    return(C)
}

# GPU Element-Wise Hyperbolic Sine
gpuVecElemHypSin <- function(A){
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    
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
    return(C)
}

# GPU Element-Wise Sine
gpuVecElemCos <- function(A){
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    
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
    return(C)
}

# GPU Element-Wise Arc Sine
gpuVecElemArcCos <- function(A){
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    
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
    return(C)
}

# GPU Element-Wise Hyperbolic Sine
gpuVecElemHypCos <- function(A){
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    
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
    return(C)
}

# GPU Element-Wise Sine
gpuVecElemTan <- function(A){
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    
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
    return(C)
}

# GPU Element-Wise Arc Sine
gpuVecElemArcTan <- function(A){
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    
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
    return(C)
}

# GPU Element-Wise Hyperbolic Sine
gpuVecElemHypTan <- function(A){
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    
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
    return(C)
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
gpuVecElemAbs <- function(A){
    
    type <- typeof(A)
    
    C <- gpuVector(length=length(A), type=type, ctx_id = A@.context_index)
    
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
    return(C)
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
