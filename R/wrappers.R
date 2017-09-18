

#' @useDynLib gpuR
#' @importFrom Rcpp evalCpp


### gpuMatrix Wrappers ###


#' @importFrom Rcpp evalCpp


# GPU axpy wrapper
gpu_Mat_axpy <- function(alpha, A, B, inplace = FALSE, AisScalar = FALSE, BisScalar = FALSE){
    
    if(inherits(A, 'gpuMatrix') & inherits(B, 'gpuMatrix')){
        assert_are_identical(A@.context_index, B@.context_index)
        
        # nrA = nrow(A)
        # ncA = ncol(A)
        # nrB = nrow(B)
        # ncB = ncol(B)
        
        type <- typeof(A)
        
    }else{
        if(inherits(A, 'gpuMatrix')){
            type <- typeof(A)
        }else{
            type <- typeof(B)
        }
    }
    
    if(inplace){
        if(!AisScalar && !BisScalar){
            Z <- B
        }else{
            if(inherits(A, 'gpuMatrix')){
                Z <- A
            }else{
                Z <- B
            }
            # if(AisScalar){
            #     Z <- B    
            # }else{
            #     Z <- A
            # }    
        }
    }else{
        if(!AisScalar && !BisScalar){
            if(!missing(B))
            {
                if(length(B[]) != length(A[])){
                    stop("Lengths of matrices must match")
                }
                Z <- deepcopy(B)
            }   
        }else{
            if(AisScalar){
                if(!missing(B))
                {
                    if(inherits(A, 'gpuMatrix')){
                        if(length(B[]) != length(A[])){
                            stop("Lengths of matrices must match")
                        }
                    }
                    Z <- deepcopy(B)
                }
            }else{
                if(!missing(A))
                {
                    if(inherits(B, 'gpuMatrix')){
                        if(length(B[]) != length(A[])){
                            stop("Lengths of matrices must match")
                        }
                    }
                    Z <- deepcopy(A)
                }     
            }
        }
        
        
    }
    
    if(AisScalar || BisScalar){
        
        scalar <- if(AisScalar) A else B
        order <- if(AisScalar) 0L else 1L
        
        # print(scalar)
        # print(alpha)
        # print(head(Z[]))
        
        maxWorkGroupSize <- 
            switch(deviceType(Z@.platform_index, Z@.device_index),
                   "gpu" = gpuInfo(Z@.platform_index, Z@.device_index)$maxWorkGroupSize,
                   "cpu" = cpuInfo(Z@.platform_index, Z@.device_index)$maxWorkGroupSize,
                   stop("unrecognized device type")
            )
        
        switch(type,
               integer = {
                   file <- system.file("CL", "iScalarAXPY.cl", package = "gpuR")
                   
                   if(!file_test("-f", file)){
                       stop("kernel file does not exist")
                   }
                   kernel <- readChar(file, file.info(file)$size)
                   
                   cpp_gpuMatrix_scalar_axpy(alpha, 
                                             scalar, 
                                             Z@address,
                                             order,
                                             sqrt(maxWorkGroupSize),
                                             kernel,
                                             Z@.context_index - 1,
                                             4L)
               },
               float = {
                   
                   file <- system.file("CL", "fScalarAXPY.cl", package = "gpuR")
                   
                   if(!file_test("-f", file)){
                       stop("kernel file does not exist")
                   }
                   kernel <- readChar(file, file.info(file)$size)
                   
                   cpp_gpuMatrix_scalar_axpy(alpha, 
                                             scalar, 
                                             Z@address,
                                             order,
                                             sqrt(maxWorkGroupSize),
                                             kernel,
                                             Z@.context_index - 1,
                                             6L)
               },
               double = {
                   
                   file <- system.file("CL", "dScalarAXPY.cl", package = "gpuR")
                   
                   if(!file_test("-f", file)){
                       stop("kernel file does not exist")
                   }
                   kernel <- readChar(file, file.info(file)$size)
                   
                   cpp_gpuMatrix_scalar_axpy(alpha, 
                                             scalar,
                                             Z@address,
                                             order,
                                             sqrt(maxWorkGroupSize),
                                             kernel,
                                             Z@.context_index - 1,
                                             8L)
               },
               stop("type not recognized")
        )
    }else{
        
        # print('default axpy')
        switch(type,
               integer = {
                   # stop("OpenCL integer GEMM not currently
                   #      supported for viennacl matrices")
                   cpp_gpuMatrix_axpy(alpha, 
                                      A@address, 
                                      Z@address,
                                      4L)
               },
               float = {
                   cpp_gpuMatrix_axpy(alpha, 
                                      A@address, 
                                      Z@address,
                                      6L)
               },
               double = {
                   cpp_gpuMatrix_axpy(alpha, 
                                      A@address,
                                      Z@address,
                                      8L)
               },
               stop("type not recognized")
        )
    }
    
    if(inplace){
        return(invisible(Z))
    }else{
        return(Z)   
    }
}

# GPU axpy wrapper
gpuMatrix_unary_axpy <- function(A, inplace = FALSE){
    
    type = typeof(A)
    
    if(inplace){
        Z <- A
    }else{
        Z <- deepcopy(A)
    }
    
    switch(type,
           integer = {
               cpp_gpuMatrix_unary_axpy(Z@address, 
                                        4L)
           },
           float = {
               cpp_gpuMatrix_unary_axpy(Z@address, 
                                        6L)
           },
           double = {
               cpp_gpuMatrix_unary_axpy(Z@address,
                                        8L)
           },
           stop("type not recognized")
    )
    
    return(Z)
}

# GPU Matrix Multiplication
gpu_Mat_mult <- function(A, B, inplace = FALSE){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- gpuMatrix(nrow=nrow(A), ncol=ncol(B), type=type, ctx_id = A@.context_index)
    }
    
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
                                          C@.context_index - 1L)
           },
           float = {cpp_gpuMatrix_gemm(A@address,
                                       B@address,
                                       C@address,
                                       6L)
           },
           double = {
               cpp_gpuMatrix_gemm(A@address,
                                  B@address,
                                  C@address,
                                  8L)
           },
           stop("type not recognized")
    )
    
    return(C)
}

# GPU Element-Wise Multiplication
gpuMatElemMult <- function(A, B, inplace = FALSE){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    if(!all(dim(A) == dim(B))){
        stop("matrices not conformable")
    }
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_prod(A@address,
                                       B@address,
                                       C@address,
                                       4L)
           },
           float = {cpp_gpuMatrix_elem_prod(A@address,
                                            B@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_gpuMatrix_elem_prod(A@address,
                                       B@address,
                                       C@address,
                                       8L)
           },
           {
               stop("type not recognized")
           })
    return(C)
}

# GPU Scalar Element-Wise Multiplication
gpuMatScalarMult <- function(A, B, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- deepcopy(A)    
    }
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_scalar_prod(C@address,
                                         B,
                                         4L)
           },
           float = {cpp_gpuMatrix_scalar_prod(C@address,
                                              B,
                                              6L)
           },
           double = {
               cpp_gpuMatrix_scalar_prod(C@address,
                                         B,
                                         8L)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Division
gpuMatElemDiv <- function(A, B, inplace = FALSE){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    if(!all(dim(A) == dim(B))){
        stop("matrices not conformable")
    }
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_div(A@address,
                                      B@address,
                                      C@address,
                                      4L)
           },
           float = {cpp_gpuMatrix_elem_div(A@address,
                                           B@address,
                                           C@address,
                                           6L)
           },
           double = {
               cpp_gpuMatrix_elem_div(A@address,
                                      B@address,
                                      C@address,
                                      8L)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Scalar Element-Wise Division
gpuMatScalarDiv <- function(A, B, AisScalar = FALSE, inplace = FALSE){
    
    if(AisScalar){
        type <- typeof(B)
        scalar <- A
        
        if(inplace){
            C <- B
        }else{
            C <- deepcopy(B)    
        }
        
        maxWorkGroupSize <- 
            switch(deviceType(C@.platform_index, C@.device_index),
                   "gpu" = gpuInfo(C@.platform_index, C@.device_index)$maxWorkGroupSize,
                   "cpu" = cpuInfo(C@.platform_index, C@.device_index)$maxWorkGroupSize,
                   stop("unrecognized device type")
            )
        
        switch(type,
               integer = {
                   src <- file <- system.file("CL", "iScalarElemDiv.cl", package = "gpuR")
                   
                   if(!file_test("-f", file)){
                       stop("kernel file does not exist")
                   }
                   kernel <- readChar(file, file.info(file)$size)
                   
                   cpp_gpuMatrix_scalar_div_2(C@address,
                                              scalar,
                                              sqrt(maxWorkGroupSize),
                                              kernel,
                                              C@.context_index - 1,
                                              4L)
               },
               float = {
                   src <- file <- system.file("CL", "fScalarElemDiv.cl", package = "gpuR")
                   
                   if(!file_test("-f", file)){
                       stop("kernel file does not exist")
                   }
                   kernel <- readChar(file, file.info(file)$size)
                   
                   cpp_gpuMatrix_scalar_div_2(C@address,
                                              scalar,
                                              sqrt(maxWorkGroupSize),
                                              kernel,
                                              C@.context_index - 1,
                                              6L)
               },
               double = {
                   src <- file <- system.file("CL", "dScalarElemDiv.cl", package = "gpuR")
                   
                   if(!file_test("-f", file)){
                       stop("kernel file does not exist")
                   }
                   kernel <- readChar(file, file.info(file)$size)
                   
                   cpp_gpuMatrix_scalar_div_2(C@address,
                                              scalar,
                                              sqrt(maxWorkGroupSize),
                                              kernel,
                                              C@.context_index - 1,
                                              8L)
               },
               stop("type not recognized")
        )
    }else{
        type <- typeof(A)
        
        if(inplace){
            C <- A
        }else{
            C <- deepcopy(A)
        }
        
        scalar <- B
        
        switch(type,
               integer = {
                   cpp_gpuMatrix_scalar_div(C@address,
                                            scalar,
                                            4L)
               },
               float = {cpp_gpuMatrix_scalar_div(C@address,
                                                 scalar,
                                                 6L)
               },
               double = {
                   cpp_gpuMatrix_scalar_div(C@address,
                                            scalar,
                                            8L)
               },
               stop("type not recognized")
        )
    }
    
    if(inplace){
        return(invisible(C))
    }else{
        return(C)    
    }
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
                                      4L)
           },
           float = {cpp_gpuMatrix_elem_pow(A@address,
                                           B@address,
                                           C@address,
                                           6L)
           },
           double = {
               cpp_gpuMatrix_elem_pow(A@address,
                                      B@address,
                                      C@address,
                                      8L)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Power
gpuMatScalarPow <- function(A, B, inplace = TRUE){
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_scalar_pow(A@address,
                                        B,
                                        C@address,
                                        4L)
           },
           float = {cpp_gpuMatrix_scalar_pow(A@address,
                                             B,
                                             C@address,
                                             6L)
           },
           double = {
               cpp_gpuMatrix_scalar_pow(A@address,
                                        B,
                                        C@address,
                                        8L)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise sqrt
gpuMatSqrt <- function(A){
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_sqrt(A@address,
                                  C@address,
                                  4L)
           },
           float = {cpp_gpuMatrix_sqrt(A@address,
                                       C@address,
                                       6L)
           },
           double = {
               cpp_gpuMatrix_sqrt(A@address,
                                  C@address,
                                  8L)
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
                                      4L)
           },
           float = {cpp_gpuMatrix_elem_sin(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               cpp_gpuMatrix_elem_sin(A@address,
                                      C@address,
                                      8L)
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
                                       4L)
           },
           float = {cpp_gpuMatrix_elem_asin(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_gpuMatrix_elem_asin(A@address,
                                       C@address,
                                       8L)
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
                                       4L)
           },
           float = {cpp_gpuMatrix_elem_sinh(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_gpuMatrix_elem_sinh(A@address,
                                       C@address,
                                       8L)
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
                                      4L)
           },
           float = {cpp_gpuMatrix_elem_cos(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               cpp_gpuMatrix_elem_cos(A@address,
                                      C@address,
                                      8L)
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
                                       4L)
           },
           float = {cpp_gpuMatrix_elem_acos(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_gpuMatrix_elem_acos(A@address,
                                       C@address,
                                       8L)
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
                                       4L)
           },
           float = {cpp_gpuMatrix_elem_cosh(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_gpuMatrix_elem_cosh(A@address,
                                       C@address,
                                       8L)
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
                                      4L)
           },
           float = {cpp_gpuMatrix_elem_tan(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               cpp_gpuMatrix_elem_tan(A@address,
                                      C@address,
                                      8L)
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
                                       4L)
           },
           float = {cpp_gpuMatrix_elem_atan(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_gpuMatrix_elem_atan(A@address,
                                       C@address,
                                       8L)
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
                                       4L)
           },
           float = {cpp_gpuMatrix_elem_tanh(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_gpuMatrix_elem_tanh(A@address,
                                       C@address,
                                       8L)
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
                                      4L)
           },
           float = {cpp_gpuMatrix_elem_log(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               cpp_gpuMatrix_elem_log(A@address,
                                      C@address,
                                      8L)
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
                                           4L)
           },
           float = {cpp_gpuMatrix_elem_log_base(A@address,
                                                C@address,
                                                base,
                                                6L)
           },
           double = {
               cpp_gpuMatrix_elem_log_base(A@address,
                                           C@address,
                                           base,
                                           8L)
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
                                        4L)
           },
           float = {cpp_gpuMatrix_elem_log10(A@address,
                                             C@address,
                                             6L)
           },
           double = {
               cpp_gpuMatrix_elem_log10(A@address,
                                        C@address,
                                        8L)
           },
           stop("type not recognized")
    )
    return(C)
}

# GPU Element-Wise Exponential
gpuMatElemExp <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_exp(A@address,
                                      C@address,
                                      4L)
           },
           float = {cpp_gpuMatrix_elem_exp(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               cpp_gpuMatrix_elem_exp(A@address,
                                      C@address,
                                      8L)
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
                                    4L)
           },
           "float" = {
               cpp_gpuMatrix_colsum(A@address, 
                                    sums@address, 
                                    6L)
           },
           "double" = {
               cpp_gpuMatrix_colsum(A@address, 
                                    sums@address, 
                                    8L)
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
                   4L)
           },
           "float" = {
               cpp_gpuMatrix_rowsum(
                   A@address, 
                   sums@address,
                   6L)
           },
           "double" = {
               cpp_gpuMatrix_rowsum(
                   A@address, 
                   sums@address,
                   8L)
           },
           stop("unsupported matrix type")
    )
    
    return(sums)
}

# GPU sum
gpuMatrix_sum <- function(A){
    
    type <- typeof(A)
    
    result <- switch(type,
                     "integer" = {
                         cpp_gpuMatrix_sum(
                             A@address,
                             4L)
                     },
                     "float" = {
                         cpp_gpuMatrix_sum(
                             A@address,
                             6L)
                     },
                     "double" = {
                         cpp_gpuMatrix_sum(
                             A@address,
                             8L)
                     },
                     stop("unsupported matrix type")
    )
    
    return(result)
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
                                     4L)
           },
           "float" = cpp_gpuMatrix_colmean(A@address, 
                                           sums@address, 
                                           6L),
           "double" = cpp_gpuMatrix_colmean(A@address, 
                                            sums@address, 
                                            8L)
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
                                     4L)
           },
           "float" = cpp_gpuMatrix_rowmean(A@address, 
                                           sums@address, 
                                           6L),
           "double" = cpp_gpuMatrix_rowmean(A@address, 
                                            sums@address, 
                                            8L)
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
               #                    4L)
           },
           "float" = cpp_gpuMatrix_pmcc(A@address, 
                                        B@address, 
                                        6L),
           "double" = cpp_gpuMatrix_pmcc(A@address, 
                                         B@address, 
                                         8L)
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
               #                         4L)
           },
           "float" = {
               cpp_gpuMatrix_crossprod(X@address, 
                                       Y@address, 
                                       Z@address,
                                       6L)
           },
           "double" = {
               cpp_gpuMatrix_crossprod(X@address, 
                                       Y@address, 
                                       Z@address, 
                                       8L)
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
               #                          4L)
           },
           "float" = {
               cpp_gpuMatrix_tcrossprod(X@address, 
                                        Y@address, 
                                        Z@address, 
                                        6L)
           },
           "double" = {
               cpp_gpuMatrix_tcrossprod(X@address,
                                        Y@address, 
                                        Z@address, 
                                        8L)
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
               #                    4L)
               },
           "float" = cpp_gpuMatrix_eucl(A@address, 
                                        D@address, 
                                        squareDist,
                                        6L),
           "double" = cpp_gpuMatrix_eucl(A@address, 
                                         D@address,
                                         squareDist,
                                         8L),
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
               #                     4L)
           },
           "float" = cpp_gpuMatrix_peucl(A@address,
                                         B@address,
                                         D@address, 
                                         squareDist, 
                                         6L),
           "double" = cpp_gpuMatrix_peucl(A@address, 
                                          B@address,
                                          D@address,
                                          squareDist,
                                          8L),
           stop("Unsupported matrix type")
    )
    
    invisible(D)
}

# GPU Element-Wise Absolute Value
gpuMatElemAbs <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               # stop("integer not currently implemented")
               cpp_gpuMatrix_elem_abs(A@address,
                                      C@address,
                                      4L)
           },
           float = {cpp_gpuMatrix_elem_abs(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               cpp_gpuMatrix_elem_abs(A@address,
                                      C@address,
                                      8L)
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
           integer = {cpp_gpuMatrix_transpose(A@address, B@address, 4L)},
           float = {cpp_gpuMatrix_transpose(A@address, B@address,  6L)},
           double = {cpp_gpuMatrix_transpose(A@address, B@address,  8L)},
           stop("type not recognized")
    )
    
    return(B)
}

# Get gpuMatrix diagonal
gpuMatrix_get_diag <- function(x){
    
    type <- typeof(x)
    
    # initialize vector to fill with diagonals
    y <- gpuVector(length = nrow(x), type = type, ctx_id = x@.context_index)
    
    switch(type,
           integer = {cpp_gpuMatrix_get_diag(x@address, y@address, 4L)},
           float = {cpp_gpuMatrix_get_diag(x@address, y@address, 6L)},
           double = {cpp_gpuMatrix_get_diag(x@address, y@address, 8L)},
           stop("type not recognized")
    )
    
    return(y)
}

# Set gpuMatrix diagonal with a gpuVector
gpuMat_gpuVec_set_diag <- function(x, value){
    
    type <- typeof(x)
    
    switch(type,
           integer = {cpp_gpuMat_gpuVec_set_diag(x@address, value@address, 4L)},
           float = {cpp_gpuMat_gpuVec_set_diag(x@address, value@address, 6L)},
           double = {cpp_gpuMat_gpuVec_set_diag(x@address, value@address, 8L)},
           stop("type not recognized")
    )
    
    return(invisible(x))
}

# GPU Determinant
gpuMat_det <- function(A){
    
    type <- typeof(A)
    
    B <- deepcopy(A)
    
    result <- switch(type,
                     integer = {
                         stop("integer not currently implemented")
                         # cpp_gpuMatrix_det(B@address, value@address, 4L)
                     },
                     float = {
                         cpp_gpuMatrix_det(B@address, 
                                           inherits(B, "vclMatrix"), 
                                           6L, 
                                           B@.context_index - 1L)},
                     double = {
                         cpp_gpuMatrix_det(B@address, 
                                           inherits(B, "vclMatrix"), 
                                           8L, 
                                           B@.context_index - 1L)},
                     stop("type not recognized")
    )
    
    return(result)
}

