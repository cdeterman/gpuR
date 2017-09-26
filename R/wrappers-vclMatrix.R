
# vclMatrix numeric vector initializer
vclMatInitNumVec <- function(data, nrow, ncol, type, ctx_id){

    device <- if(is.null(ctx_id)) currentDevice() else listContexts()[ctx_id,]
    
    context_index <- ifelse(is.null(ctx_id), currentContext(), ctx_id)
    device_index <- if(is.null(ctx_id)) as.integer(device$device_index) else device$device_index + 1L

    platform_index <- if(is.null(ctx_id)) currentPlatform()$platform_index else device$platform_index + 1L
    platform_name <- platformInfo(platform_index)$platformName
    
    device_type <- device$device_type
    device_name <- switch(device_type,
                          "gpu" = gpuInfo(
                              device_idx = as.integer(device_index),
                              context_idx = context_index)$deviceName,
                          "cpu" = cpuInfo(
                              device_idx = as.integer(device_index),
                              context_idx = context_index)$deviceName,
                          stop("Unrecognized device type")
    )
    
    data = switch(type,
                  integer = stop("integer matrix must be initialized with an integer (e.g. 3L)"),
                  float = {
                      new("fvclMatrix", 
                          address=vectorToMatVCL(data, 
                                                 nrow, ncol, 
                                                 6L, context_index - 1),
                          .context_index = context_index,
                          .platform_index = platform_index,
                          .platform = platform_name,
                          .device_index = device_index,
                          .device = device_name)
                  },
                  double = {
                      assert_has_double(device_index, context_index)
                      new("dvclMatrix",
                          address = vectorToMatVCL(data, 
                                                   nrow, ncol, 
                                                   8L, context_index - 1),
                          .context_index = context_index,
                          .platform_index = platform_index,
                          .platform = platform_name,
                          .device_index = device_index,
                          .device = device_name)
                  },
                  stop("this is an unrecognized 
                                 or unimplemented data type")
    )
    
    return(data)
}

# vclMatrix numeric initializer
vclMatInitNumScalar <- function(data, nrow, ncol, type, ctx_id){
    
    device <- if(is.null(ctx_id)) currentDevice() else listContexts()[ctx_id,]
    
    context_index <- ifelse(is.null(ctx_id), currentContext(), ctx_id)
    device_index <- if(is.null(ctx_id)) as.integer(device$device_index) else device$device_index + 1L
    
    platform_index <- if(is.null(ctx_id)) currentPlatform()$platform_index else device$platform_index + 1L
    platform_name <- platformInfo(platform_index)$platformName
    
    device_type <- device$device_type
    device_name <- switch(device_type,
                          "gpu" = gpuInfo(
                              device_idx = as.integer(device_index),
                              context_idx = context_index)$deviceName,
                          "cpu" = cpuInfo(
                              device_idx = as.integer(device_index),
                              context_idx = context_index)$deviceName,
                          stop("Unrecognized device type")
    )
    
    data = switch(type,
                  integer = stop("integer matrix must be initialized with an integer (e.g. 3L)"),
                  float = {
                      new("fvclMatrix", 
                          address=
                              cpp_scalar_vclMatrix(
                                  data, 
                                  nrow, ncol, 
                                  6L, context_index - 1),
                          .context_index = context_index,
                          .platform_index = platform_index,
                          .platform = platform_name,
                          .device_index = device_index,
                          .device = device_name
                      )
                  },
                  double = {
                      assert_has_double(device_index, context_index)
                      new("dvclMatrix",
                          address = 
                              cpp_scalar_vclMatrix(
                                  data, 
                                  nrow, ncol, 
                                  8L, context_index - 1),
                          .context_index = context_index,
                          .platform_index = platform_index,
                          .platform = platform_name,
                          .device_index = device_index,
                          .device = device_name
                      )
                  },
                  stop("this is an unrecognized 
                                 or unimplemented data type")
    )
    
    return(data)
}

# vclMatrix integer vector initializer
vclMatInitIntVec <- function(data, nrow, ncol, type, ctx_id){
    
    device <- if(is.null(ctx_id)) currentDevice() else listContexts()[ctx_id,]
    
    context_index <- ifelse(is.null(ctx_id), currentContext(), ctx_id)
    device_index <- if(is.null(ctx_id)) as.integer(device$device_index) else device$device_index + 1L
    
    platform_index <- if(is.null(ctx_id)) currentPlatform()$platform_index else device$platform_index + 1L
    platform_name <- platformInfo(platform_index)$platformName
    
    device_type <- device$device_type
    device_name <- switch(device_type,
                          "gpu" = gpuInfo(
                              device_idx = as.integer(device_index),
                              context_idx = context_index)$deviceName,
                          "cpu" = cpuInfo(
                              device_idx = as.integer(device_index),
                              context_idx = context_index)$deviceName,
                          stop("Unrecognized device type")
    )
    
    data = switch(type,
                  integer = {
                      new("ivclMatrix", 
                          address=vectorToMatVCL(data, 
                                                 nrow, ncol,
                                                 4L, context_index - 1),
                          .context_index = context_index,
                          .platform_index = platform_index,
                          .platform = platform_name,
                          .device_index = device_index,
                          .device = device_name)
                  },
                  float = {
                      new("fvclMatrix", 
                          address=vectorToMatVCL(data, 
                                                 nrow, ncol, 
                                                 6L, context_index - 1),
                          .context_index = context_index,
                          .platform_index = platform_index,
                          .platform = platform_name,
                          .device_index = device_index,
                          .device = device_name)
                  },
                  double = {
                      assert_has_double(device_index, context_index)
                      new("dvclMatrix",
                          address = vectorToMatVCL(data, 
                                                   nrow, ncol, 
                                                   8L, context_index - 1),
                          .context_index = context_index,
                          .platform_index = platform_index,
                          .platform = platform_name,
                          .device_index = device_index,
                          .device = device_name)
                  },
                  stop("this is an unrecognized 
                                     or unimplemented data type")
    )
    
    return(data)
}

# vclMatrix integer scalar initializer
vclMatInitIntScalar <- function(data, nrow, ncol, type, ctx_id){
    
    device <- if(is.null(ctx_id)) currentDevice() else listContexts()[ctx_id,]
    
    context_index <- ifelse(is.null(ctx_id), currentContext(), ctx_id)
    device_index <- if(is.null(ctx_id)) as.integer(device$device_index) else device$device_index + 1L
    
    platform_index <- if(is.null(ctx_id)) currentPlatform()$platform_index else device$platform_index + 1L
    platform_name <- platformInfo(platform_index)$platformName
    
    device_type <- device$device_type
    device_name <- switch(device_type,
                          "gpu" = gpuInfo(
                              device_idx = as.integer(device_index),
                              context_idx = context_index)$deviceName,
                          "cpu" = cpuInfo(
                              device_idx = as.integer(device_index),
                              context_idx = context_index)$deviceName,
                          stop("Unrecognized device type")
    )
    
    data = switch(type,
                  integer = {
                      new("ivclMatrix", 
                          address=
                              cpp_scalar_vclMatrix(
                                  data, 
                                  nrow, ncol, 
                                  4L, context_index - 1),
                          .context_index = context_index,
                          .platform_index = platform_index,
                          .platform = platform_name,
                          .device_index = device_index,
                          .device = device_name
                      )
                  },
                  float = {
                      new("fvclMatrix", 
                          address=
                              cpp_scalar_vclMatrix(
                                  data, 
                                  nrow, ncol, 
                                  6L, context_index - 1),
                          .context_index = context_index,
                          .platform_index = platform_index,
                          .platform = platform_name,
                          .device_index = device_index,
                          .device = device_name
                      )
                  },
                  double = {
                      assert_has_double(device_index, context_index)
                      new("dvclMatrix",
                          address = 
                              cpp_scalar_vclMatrix(
                                  data, 
                                  nrow, ncol, 
                                  8L, context_index - 1),
                          .context_index = context_index,
                          .platform_index = platform_index,
                          .platform = platform_name,
                          .device_index = device_index,
                          .device = device_name
                      )
                  },
                  stop("this is an unrecognized 
                                 or unimplemented data type")
    )
    
    return(data)
}

# vclMatrix GEMM
vclMatMult <- function(A, B){
    
    type <- typeof(A)
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(B), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               # stop("OpenCL integer GEMM not currently
               #      supported for viennacl matrices")
               
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
                                          TRUE,
                                          B@address,
                                          TRUE,
                                          C@address,
                                          TRUE,
                                          kernel,
                                          sqrt(maxWorkGroupSize),
                                          C@.context_index - 1)
           },
           float = {cpp_vclMatrix_gemm(A@address,
                                       B@address,
                                       C@address,
                                       6L)
           },
           double = {
               cpp_vclMatrix_gemm(A@address,
                                  B@address,
                                  C@address,
                                  8L)
               
           },
           stop("type not recognized")
    )
    
    return(C)
}


# vclMatrix GEMM
vclGEMV<- function(A, B){
    
    type <- typeof(A)
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    AisVec <- inherits(A, "vclVector")
    BisVec <- inherits(B, "vclVector")
    
    if(AisVec){
        C <- vclVector(length = length(A), type=type, ctx_id = A@.context_index)
    }else{
        C <- vclVector(length = length(B), type=type, ctx_id = A@.context_index)
    }
    
    
    if(AisVec){
        switch(type,
               integer = {
                   stop("OpenCL integer GEMM not currently
                        supported for viennacl matrices")
                   
                   # file <- system.file("CL", "basic_gemm.cl", package = "gpuR")
                   # 
                   # if(!file_test("-f", file)){
                   #     stop("kernel file does not exist")
                   # }
                   # kernel <- readChar(file, file.info(file)$size)
                   # 
                   # maxWorkGroupSize <- 
                   #     switch(deviceType(C@.platform_index, C@.device_index),
                   #            "gpu" = gpuInfo(C@.platform_index, C@.device_index)$maxWorkGroupSize,
                   #            "cpu" = cpuInfo(C@.platform_index, C@.device_index)$maxWorkGroupSize,
                   #            stop("unrecognized device type")
                   #     )
                   # 
                   # cpp_gpuMatrix_custom_igemm(A@address,
                   #                            TRUE,
                   #                            B@address,
                   #                            TRUE,
                   #                            C@address,
                   #                            TRUE,
                   #                            kernel,
                   #                            sqrt(maxWorkGroupSize),
                   #                            C@.context_index - 1)
               },
               float = {cpp_vclMatrix_gevm(A@address,
                                           B@address,
                                           C@address,
                                           6L)
               },
               double = {
                   cpp_vclMatrix_gevm(A@address,
                                      B@address,
                                      C@address,
                                      8L)
                   
               },
               stop("type not recognized")
        )
    }else{
        
        if(!BisVec){
            stop("B should be a vclVector object")
        }
        
        switch(type,
               integer = {
                   stop("OpenCL integer GEMM not currently
                        supported for viennacl matrices")
                   
                   # file <- system.file("CL", "basic_gemm.cl", package = "gpuR")
                   # 
                   # if(!file_test("-f", file)){
                   #     stop("kernel file does not exist")
                   # }
                   # kernel <- readChar(file, file.info(file)$size)
                   # 
                   # maxWorkGroupSize <- 
                   #     switch(deviceType(C@.platform_index, C@.device_index),
                   #            "gpu" = gpuInfo(C@.platform_index, C@.device_index)$maxWorkGroupSize,
                   #            "cpu" = cpuInfo(C@.platform_index, C@.device_index)$maxWorkGroupSize,
                   #            stop("unrecognized device type")
                   #     )
                   # 
                   # cpp_gpuMatrix_custom_igemm(A@address,
                   #                            TRUE,
                   #                            B@address,
                   #                            TRUE,
                   #                            C@address,
                   #                            TRUE,
                   #                            kernel,
                   #                            sqrt(maxWorkGroupSize),
                   #                            C@.context_index - 1)
               },
               float = {cpp_vclMatrix_gemv(A@address,
                                           B@address,
                                           C@address,
                                           6L)
               },
               double = {
                   cpp_vclMatrix_gemv(A@address,
                                      B@address,
                                      C@address,
                                      8L)
                   
               },
               stop("type not recognized")
        )
    }
    
    
    return(C)
}


# vclMatrix AXPY
vclMat_axpy <- function(alpha, A, B, inplace = FALSE, AisScalar = FALSE, BisScalar = FALSE){
    
    if(inherits(A, 'vclMatrix') & inherits(B, 'vclMatrix')){
        assert_are_identical(A@.context_index, B@.context_index)
        
        # nrA = nrow(A)
        # ncA = ncol(A)
        # nrB = nrow(B)
        # ncB = ncol(B)
        
        type <- typeof(A)
        
    }else{
        if(inherits(A, 'vclMatrix')){
            type <- typeof(A)
        }else{
            type <- typeof(B)
        }
    }
    
    if(inplace){
        if(!AisScalar && !BisScalar){
            Z <- B
        }else{
            if(inherits(A, 'vclMatrix')){
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
                    Z <- deepcopy(B)
                }
            }else{
                if(!missing(A))
                {
                    if(inherits(B, 'vclMatrix')){
                        if(length(B[]) != length(A[])){
                            stop("Lengths of matrices must match")
                        }
                    }
                    Z <- deepcopy(B)
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
                   
                   cpp_vclMatrix_scalar_axpy(alpha, 
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
                   
                   cpp_vclMatrix_scalar_axpy(alpha, 
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
                   
                   cpp_vclMatrix_scalar_axpy(alpha, 
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
                   cpp_vclMatrix_axpy(alpha, 
                                      A@address, 
                                      Z@address,
                                      4L)
               },
               float = {
                   cpp_vclMatrix_axpy(alpha, 
                                      A@address, 
                                      Z@address,
                                      6L)
               },
               double = {
                   cpp_vclMatrix_axpy(alpha, 
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


# need API for matrix-vector Arith methods
# can convert vector to 'dummy' matrix
# but the 'dummy' matrix can't be used by vclMatrix
# need to 'copy' the matrix for now because of the padding
# waiting on viennacl issues #217 & #219

vclMatVec_axpy <- function(alpha, A, B, inplace = FALSE){
 
    assert_are_identical(A@.context_index, B@.context_index)
    
    type <- typeof(A)
    
    AisVec <- inherits(A, "vclVector")
    BisVec <- inherits(B, "vclVector")
    
    if(inplace){
        Z <- B
    }else{
        
        if(AisVec != BisVec){
            
            # this is not efficient as pulling vector data from GPU and putting back
            if(AisVec){
                print("A is vector")
                Y <- vclMatrix(A[], 
                               nrow = nrow(B), 
                               ncol = ncol(B), 
                               type = type,
                               ctx_id = B@.context_index)   
                Z <- deepcopy(B)
            }else{
                print("B is vector")
                Z <- vclMatrix(B[],
                               nrow = nrow(A), 
                               ncol = ncol(A), 
                               type = type,
                               ctx_id = A@.context_index)    
                Y <- A
            }
        }else{
            Y <- A
            Z <- deepcopy(B)     
        }
    }
    
    AisVec <- inherits(Y, "vclVector")
    BisVec <- inherits(Z, "vclVector")
    
    print(Y[])
    print(Z[])
    
    # if neither vectors, then do matrix operations
    if(!AisVec & !BisVec){
        switch(type,
               integer = {
                   cpp_vclMatrix_axpy(alpha, 
                                      Y@address, 
                                      Z@address,
                                      4L)
               },
               float = {
                   cpp_vclMatrix_axpy(alpha, 
                                      Y@address, 
                                      Z@address,
                                      6L)
               },
               double = {
                   cpp_vclMatrix_axpy(alpha, 
                                      Y@address,
                                      Z@address,
                                      8L)
               },
               stop("type not recognized")
        )
    }else{
        switch(type,
               integer = {
                   cpp_vclMatVec_axpy(alpha, 
                                      Y@address, 
                                      AisVec,
                                      Z@address,
                                      BisVec,
                                      4L,
                                      Y@.context_index)
               },
               float = {
                   cpp_vclMatVec_axpy(alpha, 
                                      Y@address, 
                                      AisVec,
                                      Z@address,
                                      BisVec,
                                      6L,
                                      Y@.context_index)
               },
               double = {
                   cpp_vclMatVec_axpy(alpha, 
                                      Y@address, 
                                      AisVec,
                                      Z@address,
                                      BisVec,
                                      8L,
                                      Y@.context_index)
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


# vclMatrix unary AXPY
vclMatrix_unary_axpy <- function(A, inplace = FALSE){
    
    type = typeof(A)
    
    if(inplace){
        Z <- A
    }else{
        Z <- deepcopy(A)   
    }
    
    switch(type,
           integer = {
               cpp_vclMatrix_unary_axpy(Z@address, 
                                        4L,
                                        Z@.context_index - 1)
           },
           float = {
               cpp_vclMatrix_unary_axpy(Z@address, 
                                        6L,
                                        Z@.context_index - 1)
           },
           double = {
               cpp_vclMatrix_unary_axpy(Z@address,
                                        8L,
                                        Z@.context_index - 1)
           },
           stop("type not recognized")
    )
    
    if(inplace){
    	return(invisible(Z))
    }else{
    	return(Z)	
    }
}

# vclMatrix crossprod
vcl_crossprod <- function(X, Y){
    
    if(nrow(X) != nrow(Y)){
        stop("matrices non-conformable")
    }
    
    assert_are_identical(X@.context_index, Y@.context_index)
    
    type <- typeof(X)
    
    Z <- vclMatrix(nrow = ncol(X), ncol = ncol(Y), type = type, ctx_id = X@.context_index)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vclMatrix_crossprod(X@address, 
                                             Y@address, 
                                             Z@address,
                                             6L),
           "double" = cpp_vclMatrix_crossprod(X@address, 
                                              Y@address, 
                                              Z@address,
                                              8L)
    )
    
    return(Z)
}

# vclMatrix crossproduct where result is a vclVector
vcl_crossprod2 <- function(X, Y, Z = NULL){
    
    if(nrow(X) != nrow(Y)){
        stop("matrices non-conformable")
    }
    
    assert_are_identical(X@.context_index, Y@.context_index)
    
    type <- typeof(X)
    
    if(is.null(Z)){
        Z <- vclVector(length = ncol(X) * ncol(Y), type = type, ctx_id = X@.context_index)
        inplace = FALSE
    }else{
        if(length(Z) != ncol(X) * ncol(Y)){
            stop("dimensions don't match")
        }
        inplace = TRUE
    }
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vclMat_vclVec_crossprod(X@address, 
                                             Y@address, 
                                             Z@address,
                                             6L),
           "double" = cpp_vclMat_vclVec_crossprod(X@address, 
                                              Y@address, 
                                              Z@address,
                                              8L)
    )
    
    if(inplace){
        return(invisible(Z))
    }else{
        return(Z)    
    }
}

# vclMatrix-vclVector crossproduct
vcl_mat_vec_crossprod <- function(X, Y, Z = NULL){
    
    assert_are_identical(X@.context_index, Y@.context_index)
    
    type <- typeof(X)
    
    AisVec <- inherits(X, "vclVector")
    BisVec <- inherits(Y, "vclVector")
    
    if(AisVec){
        if(length(X) != nrow(Y)){
            stop("non-conformable arguments")
        }
    }else{
        if(nrow(X) != length(Y)){
            stop("non-conformable arguments")
        }
    }
    
    if(is.null(Z)){
        if(AisVec){
            Z <- vclVector(length = ncol(Y), type = type, ctx_id = X@.context_index)
        }else{
            Z <- vclVector(length = ncol(X), type = type, ctx_id = X@.context_index)    
        }
        inplace = FALSE
    }else{
        if(AisVec){
            if(length(Z) != ncol(Y)){
                stop("dimensions don't match")
            }
        }else{
            if(length(Z) != ncol(X)){
                stop("dimensions don't match")
            }
        }
        
        inplace = TRUE
    }
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vclMatVec_crossprod(X@address, 
                                             AisVec,
                                             Y@address, 
                                             BisVec,
                                             Z@address,
                                             6L),
           "double" = cpp_vclMatVec_crossprod(X@address, 
                                              AisVec,
                                              Y@address, 
                                              BisVec,
                                              Z@address,
                                              8L)
    )
    
    if(inplace){
        return(invisible(Z))
    }else{
        return(Z)    
    }
}

# vclMatrix crossprod
vcl_tcrossprod <- function(X, Y){
    
    if(ncol(X) != ncol(Y)){
        stop("matrices non-conformable")
    }
    
    assert_are_identical(X@.context_index, Y@.context_index)
    
    type <- typeof(X)
    
    Z <- vclMatrix(nrow = nrow(X), ncol = nrow(Y), type = type, ctx_id=X@.context_index)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vclMatrix_tcrossprod(X@address,
                                              Y@address, 
                                              Z@address,
                                              6L),
           "double" = cpp_vclMatrix_tcrossprod(X@address, 
                                               Y@address, 
                                               Z@address,
                                               8L),
           stop("type not recognized")
    )
    
    return(Z)
}

# vclMatrix-vclVector tcrossproduct
vcl_mat_vec_tcrossprod <- function(X, Y, Z = NULL){
    
    assert_are_identical(X@.context_index, Y@.context_index)
    
    type <- typeof(X)
    
    AisVec <- inherits(X, "vclVector")
    BisVec <- inherits(Y, "vclVector")
    
    if(AisVec){
        if(length(X) != ncol(Y)){
            if(ncol(Y) != 1){
                stop("non-conformable arguments")   
            }
        }
    }else{
        if(ncol(X) != 1){
            stop("non-conformable arguments")
        }
    }
    
    if(is.null(Z)){
        if(AisVec){
            if(ncol(Y) == 1){
                Z <- vclMatrix(nrow = length(X), ncol = nrow(Y), type = type, ctx_id = X@.context_index)   
                CisVec <- FALSE    
            }else{
                Z <- vclVector(length = nrow(Y), type = type, ctx_id = X@.context_index)
                CisVec <- TRUE
            }
        }else{
            if(ncol(X) == 1){
                Z <- vclMatrix(nrow = nrow(X), ncol = length(Y), type = type, ctx_id = X@.context_index)   
                CisVec <- FALSE    
            }else{
                Z <- vclVector(length = nrow(X), type = type, ctx_id = X@.context_index)
                CisVec <- TRUE
            }
            
        }
        inplace = FALSE
    }else{
        
        # to be tested
        if(AisVec){
        	if(ncol(Y) == 1){
        		if(nrow(Z) != length(X) || ncol(Z) != nrow(Y)){
        			stop("Output matrix not conformant to arguments")
        		}
        		CisVec <- FALSE    
        	}else{
        		if(length(Z) != ncol(Y)){
        			stop("dimensions don't match")
        		}
        		CisVec <- TRUE
        	}
        }else{
        	if(ncol(X) == 1){
	            stop("still need to troubleshoot this")
        		CisVec <- FALSE
        	}else{
        		if(length(Z) != ncol(X)){
        			stop("dimensions don't match")
        		}
        		CisVec <- TRUE
        	}
        }
        
        inplace = TRUE
    }
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vclMatVec_tcrossprod(X@address, 
                                             AisVec,
                                             Y@address, 
                                             BisVec,
                                             Z@address,
                                             CisVec,
                                             6L),
           "double" = cpp_vclMatVec_tcrossprod(X@address, 
                                              AisVec,
                                              Y@address, 
                                              BisVec,
                                              Z@address,
                                              CisVec,
                                              8L)
    )
    
    if(inplace){
        return(invisible(Z))
    }else{
        return(Z)    
    }
}

# GPU Element-Wise Multiplication
vclMatElemMult <- function(A, B, inplace = FALSE){
    
    if(!all(dim(A) == dim(B))){
        stop("matrices not conformable")
    }
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)   
    }
    
    switch(type,
           integer = {
               cpp_vclMatrix_elem_prod(A@address,
                                       B@address,
                                       C@address,
                                       4L)
           },
           float = {cpp_vclMatrix_elem_prod(A@address,
                                            B@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_vclMatrix_elem_prod(A@address,
                                       B@address,
                                       C@address,
                                       8L)
           },
           stop("type not recognized")
    )
    
    return(C)
}

# GPU Scalar Element-Wise Multiplication
vclMatScalarMult <- function(A, B, inplace = FALSE){
    
    type <- typeof(A)

    if(inplace){
        C <- A
    }else{
        C <- deepcopy(A)    
    }
    
    switch(type,
           integer = {
               cpp_vclMatrix_scalar_prod(C@address,
                                         B,
                                         4L)
           },
           float = {cpp_vclMatrix_scalar_prod(C@address,
                                              B,
                                              6L)
           },
           double = {
               cpp_vclMatrix_scalar_prod(C@address,
                                         B,
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

# GPU Element-Wise Division
vclMatElemDiv <- function(A, B, inplace = FALSE){
    
    if(!all(dim(A) == dim(B))){
        stop("matrices not conformable")
    }
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)   
    }
    
    switch(type,
           integer = {
               cpp_vclMatrix_elem_div(A@address,
                                      B@address,
                                      C@address,
                                      4L)
           },
           float = {cpp_vclMatrix_elem_div(A@address,
                                           B@address,
                                           C@address,
                                           6L)
           },
           double = {
               cpp_vclMatrix_elem_div(A@address,
                                      B@address,
                                      C@address,
                                      8L)
           },           
           stop("type not recognized")
    )
    
    return(C)
}

# GPU Scalar Element-Wise Division
vclMatScalarDiv <- function(A, B, AisScalar = FALSE, inplace = FALSE){
    
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
                   
                   cpp_vclMatrix_scalar_div_2(C@address,
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
                   
                   cpp_vclMatrix_scalar_div_2(C@address,
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
                   
                   cpp_vclMatrix_scalar_div_2(C@address,
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
                   cpp_vclMatrix_scalar_div(C@address,
                                            scalar,
                                            4L)
               },
               float = {cpp_vclMatrix_scalar_div(C@address,
                                                 scalar,
                                                 6L)
               },
               double = {
                   cpp_vclMatrix_scalar_div(C@address,
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
vclMatElemPow <- function(A, B){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    if(!all(dim(A) == dim(B))){
        stop("matrices not conformable")
    }
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               cpp_vclMatrix_elem_pow(A@address,
                                      B@address,
                                      C@address,
                                      4L)
           },
           float = {cpp_vclMatrix_elem_pow(A@address,
                                           B@address,
                                           C@address,
                                           6L)
           },
           double = {
               cpp_vclMatrix_elem_pow(A@address,
                                      B@address,
                                      C@address,
                                      8L)
           },
           stop("type not recognized")
    )
    
    return(C)
}

# GPU Element-Wise Power
vclMatScalarPow <- function(A, B){
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               cpp_vclMatrix_scalar_pow(A@address,
                                        B,
                                        C@address,
                                        4L,
                                        A@.context_index - 1)
           },
           float = {cpp_vclMatrix_scalar_pow(A@address,
                                             B,
                                             C@address,
                                             6L,
                                             A@.context_index - 1)
           },
           double = {
               cpp_vclMatrix_scalar_pow(A@address,
                                        B,
                                        C@address,
                                        8L,
                                        A@.context_index - 1)
           },
           stop("type not recognized")
    )
 
    return(C)
}

# GPU Element-Wise sqrt
vclMatSqrt <- function(A){
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               cpp_vclMatrix_sqrt(A@address,
                                  C@address,
                                  4L)
           },
           float = {cpp_vclMatrix_sqrt(A@address,
                                       C@address,
                                       6L)
           },
           double = {
               cpp_vclMatrix_sqrt(A@address,
                                  C@address,
                                  8L)
           },
           stop("type not recognized")
    )
    
    return(C)
}

# GPU Element-Wise Sine
vclMatElemSin <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)    
    }
    
    
    switch(type,
           integer = {
               cpp_vclMatrix_elem_sin(A@address,
                                      C@address,
                                      4L)
           },
           float = {cpp_vclMatrix_elem_sin(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               cpp_vclMatrix_elem_sin(A@address,
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
vclMatElemArcSin <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)    
    }
    
    switch(type,
           integer = {
               cpp_vclMatrix_elem_asin(A@address,
                                       C@address,
                                       4L)
           },
           float = {cpp_vclMatrix_elem_asin(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_vclMatrix_elem_asin(A@address,
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
vclMatElemHypSin <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A   
    }else{
        C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)    
    }
    
    switch(type,
           integer = {
               cpp_vclMatrix_elem_sinh(A@address,
                                       C@address,
                                       4L)
           },
           float = {cpp_vclMatrix_elem_sinh(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_vclMatrix_elem_sinh(A@address,
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
vclMatElemCos <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)    
    }
    
    switch(type,
           integer = {
               cpp_vclMatrix_elem_cos(A@address,
                                      C@address,
                                      4L)
           },
           float = {cpp_vclMatrix_elem_cos(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               cpp_vclMatrix_elem_cos(A@address,
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
vclMatElemArcCos <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    }
    
    switch(type,
           integer = {
               cpp_vclMatrix_elem_acos(A@address,
                                       C@address,
                                       4L)
           },
           float = {cpp_vclMatrix_elem_acos(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_vclMatrix_elem_acos(A@address,
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
vclMatElemHypCos <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)    
    }
    
    switch(type,
           integer = {
               cpp_vclMatrix_elem_cosh(A@address,
                                       C@address,
                                       4L)
           },
           float = {cpp_vclMatrix_elem_cosh(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_vclMatrix_elem_cosh(A@address,
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
vclMatElemTan <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)    
    }
    
    switch(type,
           integer = {
               cpp_vclMatrix_elem_tan(A@address,
                                      C@address,
                                      4L)
           },
           float = {cpp_vclMatrix_elem_tan(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               cpp_vclMatrix_elem_tan(A@address,
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
vclMatElemArcTan <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)    
    }
    
    
    switch(type,
           integer = {
               cpp_vclMatrix_elem_atan(A@address,
                                       C@address,
                                       4L)
           },
           float = {cpp_vclMatrix_elem_atan(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_vclMatrix_elem_atan(A@address,
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
vclMatElemHypTan <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    }
    
    
    switch(type,
           integer = {
               cpp_vclMatrix_elem_tanh(A@address,
                                       C@address,
                                       4L)
           },
           float = {cpp_vclMatrix_elem_tanh(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               cpp_vclMatrix_elem_tanh(A@address,
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
vclMatElemLog <- function(A){
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               cpp_vclMatrix_elem_log(A@address,
                                      C@address,
                                      4L)
           },
           float = {cpp_vclMatrix_elem_log(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               cpp_vclMatrix_elem_log(A@address,
                                      C@address,
                                      8L)
           },
           stop("type not recognized")
    )
    
    return(C)
}

# GPU Element-Wise Log Base
vclMatElemLogBase <- function(A, base){
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               cpp_vclMatrix_elem_log_base(A@address,
                                           C@address,
                                           base,
                                           4L)
           },
           float = {cpp_vclMatrix_elem_log_base(A@address,
                                                C@address,
                                                base,
                                                6L)
           },
           double = {
               cpp_vclMatrix_elem_log_base(A@address,
                                           C@address,
                                           base,
                                           8L)
           },
           stop("type not recognized")
    )
    
    return(C)
}

# GPU Element-Wise Base 10 Log
vclMatElemLog10 <- function(A){
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    
    switch(type,
           integer = {
               cpp_vclMatrix_elem_log10(A@address,
                                        C@address,
                                        4L)
           },
           float = {cpp_vclMatrix_elem_log10(A@address,
                                             C@address,
                                             6L)
           },
           double = {
               cpp_vclMatrix_elem_log10(A@address,
                                        C@address,
                                        8L)
           },
           stop("type not recognized")
    )
    
    return(C)
}

# GPU Element-Wise Exponential
vclMatElemExp <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)   
    }
    
    switch(type,
           integer = {
               cpp_vclMatrix_elem_exp(A@address,
                                      C@address,
                                      4L)
           },
           float = {cpp_vclMatrix_elem_exp(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               cpp_vclMatrix_elem_exp(A@address,
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

# vclMatrix colSums
vclMatrix_colSums <- function(A){
    
    type <- typeof(A)
    
    if(type == "integer"){
        stop("integer type not currently implemented")
    }
    
    sums <- vclVector(length = ncol(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vclMatrix_colsum(A@address, 
                                          sums@address, 
                                          6L),
           "double" = cpp_vclMatrix_colsum(A@address, 
                                           sums@address, 
                                           8L),
           stop("unsupported matrix type")
    )
    
    return(sums)
}

# vclMatrix rowSums
vclMatrix_rowSums <- function(A){
    
    type <- typeof(A)
    
    if(type == "integer"){
        stop("integer type not currently implemented")
    }
    
    sums <- vclVector(length = nrow(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vclMatrix_rowsum(A@address, 
                                          sums@address, 
                                          6L),
           "double" = cpp_vclMatrix_rowsum(A@address, 
                                           sums@address, 
                                           8L),
           stop("unsupported matrix type")
    )
    
    return(sums)
}

# vclMatrix sum
vclMatSum <- function(A){
    
    type <- typeof(A)
    
    result <- switch(type,
           "integer" = cpp_vclMatrix_sum(A@address, 
                                         4L),
           "float" = cpp_vclMatrix_sum(A@address, 
                                       6L),
           "double" = cpp_vclMatrix_sum(A@address,
                                        8L),
           stop("unsupported matrix type")
    )
    
    return(result)
}

# vclMatrix colMeans
vclMatrix_colMeans <- function(A){
    
    type <- typeof(A)
    
    if(type == "integer"){
        stop("integer type not currently implemented")
    }
    
    sums <- vclVector(length = ncol(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vclMatrix_colmean(A@address, 
                                           sums@address, 
                                           6L),
           "double" = cpp_vclMatrix_colmean(A@address, 
                                            sums@address, 
                                            8L),
           stop("unsupported matrix type")
    )
    
    return(sums)
}

# vclMatrix rowMeans
vclMatrix_rowMeans <- function(A){
    
    type <- typeof(A)
    
    if(type == "integer"){
        stop("integer type not currently implemented")
    }
    
    sums <- vclVector(length = nrow(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vclMatrix_rowmean(A@address, 
                                           sums@address,
                                           6L),
           "double" = cpp_vclMatrix_rowmean(A@address, 
                                            sums@address, 
                                            8L),
           stop("unsupported matrix type")
    )
    
    return(sums)
}

# GPU Pearson Covariance
vclMatrix_pmcc <- function(A, B){
    
    type <- typeof(A)
    
    if(missing(B)){
        B <- vclMatrix(nrow = ncol(A), ncol = ncol(A), type = type, ctx_id = A@.context_index)
        
        switch(type,
               "integer" = stop("integer type not currently implemented"),
               "float" = cpp_vclMatrix_pmcc(A@address, 
                                            B@address, 
                                            6L,
                                            A@.context_index - 1),
               "double" = cpp_vclMatrix_pmcc(A@address, 
                                             B@address,
                                             8L,
                                             A@.context_index - 1),
               stop("unsupported matrix type")
        )
        
        return(B)
    }else{
        
        assert_are_identical(A@.context_index, B@.context_index)
        
        C <- vclMatrix(nrow = ncol(A), ncol = ncol(B), type = type, ctx_id = A@.context_index)
        
        switch(type,
               "integer" = stop("integer type not currently implemented"),
               "float" = cpp_vclMatrix_pmcc2(A@address, 
                                            B@address, 
                                            C@address,
                                            6L,
                                            A@.context_index - 1),
               "double" = cpp_vclMatrix_pmcc2(A@address, 
                                             B@address,
                                             C@address,
                                             8L,
                                             A@.context_index - 1),
               stop("unsupported matrix type")
        )
        
        return(C)
    }
    
}

# GPU Euclidean Distance
vclMatrix_euclidean <- function(A, D, diag, upper, p, squareDist){
    
    assert_are_identical(A@.context_index, D@.context_index)
    
    type <- typeof(D)
    
    switch(type,
           "integer" = {
               stop("integer method not currently implemented")
               # cpp_vclMatrix_eucl(A@address, 
               #                    D@address, 
               #                    squareDist, 
               #                    4L,
               #                    A@.context_index - 1)
               },
           "float" = cpp_vclMatrix_eucl(A@address, 
                                        D@address, 
                                        squareDist, 
                                        6L,
					A@.context_index - 1),
           "double" = cpp_vclMatrix_eucl(A@address, 
                                         D@address,
                                         squareDist,
                                         8L,
					 A@.context_index - 1),
           stop("Unsupported matrix type")
    )
    
    invisible(D)
}

# GPU Pairwise Euclidean Distance
vclMatrix_peuclidean <- function(A, B, D, squareDist){
    
    if(length(unique(c(A@.context_index, B@.context_index, D@.context_index))) != 1){
        stop("Context indices don't match between arguments")
    }
    
    type <- typeof(D)
    
    switch(type,
           "integer" = {
               stop("integer method not currently implemented")
               # cpp_vclMatrix_peucl(A@address,
               #                     B@address,
               #                     D@address, 
               #                     squareDist, 
               #                     4L,
               #                     A@.context_index - 1)
               },
           "float" = cpp_vclMatrix_peucl(A@address,
                                         B@address,
                                        D@address, 
                                        squareDist, 
                                        6L,
					A@.context_index - 1),
           "double" = cpp_vclMatrix_peucl(A@address, 
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
vclMatElemAbs <- function(A, inplace = FALSE){
    
    type <- typeof(A)
    
    if(inplace){
        C <- A
    }else{
        C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)    
    }
    
    switch(type,
           integer = {
               cpp_vclMatrix_elem_abs(A@address,
                                      C@address,
                                      4L)
           },
           float = {cpp_vclMatrix_elem_abs(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               cpp_vclMatrix_elem_abs(A@address,
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

# GPU Vector maximum
vclMatMax <- function(A){
    
    type <- typeof(A)
    
    C <- switch(type,
                integer = {
                    cpp_vclMatrix_max(A@address, 4L)
                },
                float = {cpp_vclMatrix_max(A@address, 6L)
                },
                double = {
                    cpp_vclMatrix_max(A@address, 8L)
                },
                stop("type not recognized")
    )
    
    return(C)
}

# GPU Vector minimum
vclMatMin <- function(A){
    
    type <- typeof(A)
    
    C <- switch(type,
                integer = {
                    cpp_vclMatrix_min(A@address, 4L)
                },
                float = {cpp_vclMatrix_min(A@address, 6L)
                },
                double = {
                    cpp_vclMatrix_min(A@address, 8L)
                },
                stop("type not recognized")
    )
    
    return(C)
}

# GPU Matrix transpose
vclMatrix_t <- function(A){
    
    type <- typeof(A)
    
    B <- vclMatrix(0, ncol = nrow(A), nrow = ncol(A), type = type, ctx_id=A@.context_index)

    switch(type,
           integer = {cpp_vclMatrix_transpose(A@address, B@address, 4L)},
           float = {cpp_vclMatrix_transpose(A@address, B@address, 6L)},
           double = {cpp_vclMatrix_transpose(A@address, B@address, 8L)},
           stop("type not recognized")
    )
    
    return(B)
}


vclMatrix_get_diag <- function(x){
    
    type <- typeof(x)
    
    # initialize vector to fill with diagonals
    y <- vclVector(length = nrow(x), type = type, ctx_id = x@.context_index)
    
    switch(type,
           integer = {cpp_vclMatrix_get_diag(x@address, y@address, 4L)},
           float = {cpp_vclMatrix_get_diag(x@address, y@address, 6L)},
           double = {cpp_vclMatrix_get_diag(x@address, y@address, 8L)},
           stop("type not recognized")
    )
    
    return(y)
}


vclMat_vclVec_set_diag <- function(x, value){
    
    type <- typeof(x)
    
    switch(type,
           integer = {cpp_vclMat_vclVec_set_diag(x@address, value@address, 4L)},
           float = {cpp_vclMat_vclVec_set_diag(x@address, value@address, 6L)},
           double = {cpp_vclMat_vclVec_set_diag(x@address, value@address, 8L)},
           stop("type not recognized")
    )
    
    return(invisible(x))
}


