
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
                   switch(deviceType(C@.device_index, C@.context_index),
                          "gpu" = gpuInfo(C@.device_index, C@.context_index)$maxWorkGroupSize,
                          "cpu" = cpuInfo(C@.device_index, C@.context_index)$maxWorkGroupSize,
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
        C <- vclVector(length = nrow(B), type=type, ctx_id = A@.context_index)
    }else{
        C <- vclVector(length = nrow(A), type=type, ctx_id = A@.context_index)
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


# need API for matrix-vector Arith methods
# can convert vector to 'dummy' matrix
# but the 'dummy' matrix can't be used by vclMatrix
# need to 'copy' the matrix for now because of the padding
# waiting on viennacl issues #217 & #219

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


