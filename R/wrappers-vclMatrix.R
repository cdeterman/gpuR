
# vclMatrix numeric vector initializer
vclMatInitNumVec <- function(data, nrow, ncol, type){

    device <- currentDevice()
    
    context_index <- currentContext()
    device_index <- device$device_index
    device_type <- device$device_type
    device_name <- switch(device_type,
                          "gpu" = gpuInfo(device_idx = as.integer(device_index))$deviceName,
                          "cpu" = cpuInfo(device_idx = as.integer(device_index))$deviceName,
                          stop("Unrecognized device type")
    )
    platform_index <- currentPlatform()$platform_index
    platform_name <- platformInfo(platform_index)$platformName
    
    if(type == "double" & !deviceHasDouble(platform_index, device_index)){
        stop("Double precision not supported for current device. 
                       Try setting 'type = 'float'' or change device if multiple available.")
    }

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
vclMatInitNumScalar <- function(data, nrow, ncol, type){
    
    device <- currentDevice()
    
    context_index <- currentContext()
    device_index <- device$device_index
    device_type <- device$device_type
    device_name <- switch(device_type,
                          "gpu" = gpuInfo(device_idx = as.integer(device_index))$deviceName,
                          "cpu" = cpuInfo(device_idx = as.integer(device_index))$deviceName,
                          stop("Unrecognized device type")
    )
    platform_index <- currentPlatform()$platform_index
    platform_name <- platformInfo(platform_index)$platformName
    
    if(type == "double" & !deviceHasDouble(platform_index, device_index)){
        stop("Double precision not supported for current device. 
                       Try setting 'type = 'float'' or change device if multiple available.")
    }
    
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
vclMatInitIntVec <- function(data, nrow, ncol, type){
    
    device <- currentDevice()
    
    context_index <- currentContext()
    device_index <- device$device_index
    device_type <- device$device_type
    device_name <- switch(device_type,
                          "gpu" = gpuInfo(device_idx = as.integer(device_index))$deviceName,
                          "cpu" = cpuInfo(device_idx = as.integer(device_index))$deviceName,
                          stop("Unrecognized device type")
    )
    platform_index <- currentPlatform()$platform_index
    platform_name <- platformInfo(platform_index)$platformName
    
    if(type == "double" & !deviceHasDouble(platform_index, device_index)){
        stop("Double precision not supported for current device. 
                       Try setting 'type = 'float'' or change device if multiple available.")
    }
    
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
vclMatInitIntScalar <- function(data, nrow, ncol, type){
    
    device <- currentDevice()
    
    context_index <- currentContext()
    device_index <- device$device_index
    device_type <- device$device_type
    device_name <- switch(device_type,
                          "gpu" = gpuInfo(device_idx = as.integer(device_index))$deviceName,
                          "cpu" = cpuInfo(device_idx = as.integer(device_index))$deviceName,
                          stop("Unrecognized device type")
    )
    platform_index <- currentPlatform()$platform_index
    platform_name <- platformInfo(platform_index)$platformName
    
    if(type == "double" & !deviceHasDouble(platform_index, device_index)){
        stop("Double precision not supported for current device. 
                       Try setting 'type = 'float'' or change device if multiple available.")
    }
    
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
    
#     pkg_path <- find.package("gpuR", .libPaths())
#     file <- file.path(pkg_path, "CL", "basic_gemm.cl")
#     
#     if(!file_test("-f", file)){
#         stop("kernel file does not exist")
#     }
#     kernel <- readChar(file, file.info(file)$size)
    
    type <- typeof(A)
    
    assert_are_identical(A@.context_index, B@.context_index)
    
#     oldContext <- currentContext()
#     if(oldContext != A@.context_index){
#         setContext(A@.context_index)
#     }
    
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
           float = {cpp_vclMatrix_gemm(A@address,
                                       B@address,
                                       C@address,
                                       6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_gemm(A@address,
                                        B@address,
                                        C@address,
                                        8L)
               }
           },
           stop("type not recognized")
    )
    
#     if(oldContext != A@.context_index){
#         setContext(oldContext)
#     }
    
    return(C)
}

# vclMatrix AXPY
vclMat_axpy <- function(alpha, A, B){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
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
        if(length(B[]) != length(A[])){
		if(oldContext != A@.context_index) setContext(oldContext)
		stop("Lengths of matrices must match")
	i}
        Z <- deepcopy(B)
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
           float = {cpp_vclMatrix_axpy(alpha, 
                                       A@address, 
                                       Z@address,
                                       6L)
           },
           double = {cpp_vclMatrix_axpy(alpha, 
                                        A@address,
                                        Z@address,
                                        8L)
           },
            stop("type not recognized")
    )

    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(Z)
}

# vclMatrix unary AXPY
vclMatrix_unary_axpy <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type = typeof(A)
    
    Z <- deepcopy(A)
    
    switch(type,
           integer = {
               cpp_vclMatrix_unary_axpy(Z@address, 
                                        4L)
           },
           float = {
               cpp_vclMatrix_unary_axpy(Z@address, 
                                        6L)
           },
           double = {
               cpp_vclMatrix_unary_axpy(Z@address,
                                        8L)
           },
           stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(Z)
}

# vclMatrix crossprod
vcl_crossprod <- function(X, Y){
    
    if(nrow(X) != nrow(Y)){
        stop("matrices non-conformable")
    }
    
    assert_are_identical(X@.context_index, Y@.context_index)
    
    oldContext <- currentContext()
    if(oldContext != X@.context_index){
        setContext(X@.context_index)
    }
    
    type <- typeof(X)
    
    Z <- vclMatrix(nrow = ncol(X), ncol = ncol(Y), type = type)
    
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
    
    if(oldContext != X@.context_index){
        setContext(oldContext)
    }
    
    return(Z)
}

# vclMatrix crossprod
vcl_tcrossprod <- function(X, Y){
    
    if(ncol(X) != ncol(Y)){
        stop("matrices non-conformable")
    }
    
    assert_are_identical(X@.context_index, Y@.context_index)
    
    oldContext <- currentContext()
    if(oldContext != X@.context_index){
        setContext(X@.context_index)
    }
    
    type <- typeof(X)
    
    Z <- vclMatrix(nrow = nrow(X), ncol = nrow(Y), type = type)
    
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
    
    if(oldContext != X@.context_index){
        setContext(oldContext)
    }
    
    return(Z)
}


# GPU Element-Wise Multiplication
vclMatElemMult <- function(A, B){
    
    if(!all(dim(A) == dim(B))){
        stop("matrices not conformable")
    }
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclMatrix_elem_prod(A@address,
                                            B@address,
                                            C@address,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_elem_prod(A@address,
                                             B@address,
                                             C@address,
                                             8L)
               }
           },
           stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(C)
}

# GPU Scalar Element-Wise Multiplication
vclMatScalarMult <- function(A, B){
    
    type <- typeof(A)

    oldContext <- currentContext()

    if(is(A, "vclMatrix")){
	    if(oldContext != A@.context_index){
		setContext(A@.context_index)
	    }
    }else{
	    if(oldContext != B@.context_index){
		setContext(B@.context_index)
	    }

    }
    
    C <- deepcopy(A)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclMatrix_scalar_prod(C@address,
                                              B,
                                              6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_scalar_prod(C@address,
                                               B,
                                               8L)
               }
           },
           stop("type not recognized")
    )

    if(is(A, "vclMatrix")){
	    if(oldContext != A@.context_index){
		setContext(oldContext)
	    }
    }else{
	    if(oldContext != B@.context_index){
		setContext(oldContext)
	    }

    }

    return(C)
}

# GPU Element-Wise Division
vclMatElemDiv <- function(A, B){
    
    if(!all(dim(A) == dim(B))){
        stop("matrices not conformable")
    }
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclMatrix_elem_div(A@address,
                                           B@address,
                                           C@address,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_elem_div(A@address,
                                            B@address,
                                            C@address,
                                            8L)
               }
           },           
           stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(C)
}

# GPU Scalar Element-Wise Division
vclMatScalarDiv <- function(A, B){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    if(is(A, "vclMatrix")){
	    if(oldContext != A@.context_index){
		setContext(A@.context_index)
	    }
    }else{
	    if(oldContext != B@.context_index){
		setContext(B@.context_index)
	    }

    }


    
    type <- typeof(A)
    
    C <- deepcopy(A)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclMatrix_scalar_div(C@address,
                                             B,
                                             6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_scalar_div(C@address,
                                              B,
                                              8L)
               }
           },
           stop("type not recognized")
    )

    if(is(A, "vclMatrix")){
	    if(oldContext != A@.context_index){
		setContext(oldContext)
	    }
    }else{
	    if(oldContext != B@.context_index){
		setContext(oldContext)
	    }

    }

    return(C)
}

# GPU Element-Wise Power
vclMatElemPow <- function(A, B){
    
    assert_are_identical(A@.context_index, B@.context_index)
    
    if(!all(dim(A) == dim(B))){
        stop("matrices not conformable")
    }
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    

    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclMatrix_elem_pow(A@address,
                                           B@address,
                                           C@address,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_elem_pow(A@address,
                                            B@address,
                                            C@address,
                                            8L)
               }
           },
           stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(C)
}

# GPU Element-Wise Power
vclMatScalarPow <- function(A, B){
    
    oldContext <- currentContext()

    if(is(A, "vclMatrix")){
	    if(oldContext != A@.context_index){
		setContext(A@.context_index)
	    }
    }else{
	    if(oldContext != B@.context_index){
		setContext(B@.context_index)
	    }

    }
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclMatrix_scalar_pow(A@address,
                                             B,
                                             C@address,
                                             6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_scalar_pow(A@address,
                                              B,
                                              C@address,
                                              8L)
               }
           },
           stop("type not recognized")
    )
 
    if(is(A, "vclMatrix")){
	    if(oldContext != A@.context_index){
		setContext(oldContext)
	    }
    }else{
	    if(oldContext != B@.context_index){
		setContext(oldContext)
	    }

    }   
    
    return(C)
}

# GPU Element-Wise Sine
vclMatElemSin <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclMatrix_elem_sin(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_elem_sin(A@address,
                                            C@address,
                                            8L)
               }
           },
           stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(C)
}

# GPU Element-Wise Arc Sine
vclMatElemArcSin <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclMatrix_elem_asin(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_elem_asin(A@address,
                                             C@address,
                                             8L)
               }
           },
           stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(C)
}

# GPU Element-Wise Hyperbolic Sine
vclMatElemHypSin <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclMatrix_elem_sinh(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_elem_sinh(A@address,
                                             C@address,
                                             8L)
               }
           },
           
           stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    return(C)
}

# GPU Element-Wise Cos
vclMatElemCos <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclMatrix_elem_cos(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_elem_cos(A@address,
                                            C@address,
                                            8L)
               }
           },
           stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(C)
}

# GPU Element-Wise Arc Cos
vclMatElemArcCos <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclMatrix_elem_acos(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_elem_acos(A@address,
                                             C@address,
                                             8L)
               }
           },
           stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(C)
}

# GPU Element-Wise Hyperbolic Cos
vclMatElemHypCos <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclMatrix_elem_cosh(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_elem_cosh(A@address,
                                             C@address,
                                             8L)
               }
           },
           stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(C)
}

# GPU Element-Wise Tan
vclMatElemTan <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclMatrix_elem_tan(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_elem_tan(A@address,
                                            C@address,
                                            8L)
               }
           },
           stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(C)
}

# GPU Element-Wise Arc Tan
vclMatElemArcTan <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclMatrix_elem_atan(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_elem_atan(A@address,
                                             C@address,
                                             8L)
               }
           },
           stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(C)
}

# GPU Element-Wise Hyperbolic Tan
vclMatElemHypTan <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclMatrix_elem_tanh(A@address,
                                            C@address,
                                            6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_elem_tanh(A@address,
                                             C@address,
                                             8L)
               }
           },
           stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(C)
}

# GPU Element-Wise Natural Log
vclMatElemLog <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclMatrix_elem_log(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_elem_log(A@address,
                                            C@address,
                                            8L)
               }
           },
           stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(C)
}

# GPU Element-Wise Log Base
vclMatElemLogBase <- function(A, base){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclMatrix_elem_log_base(A@address,
                                                C@address,
                                                base,
                                                6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_elem_log_base(A@address,
                                                 C@address,
                                                 base,
                                                 8L)
               }
           },
           stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(C)
}

# GPU Element-Wise Base 10 Log
vclMatElemLog10 <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclMatrix_elem_log10(A@address,
                                             C@address,
                                             6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_elem_log10(A@address,
                                              C@address,
                                              8L)
               }
           },
           stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(C)
}

# GPU Element-Wise Exponential
vclMatElemExp <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclMatrix_elem_exp(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_elem_exp(A@address,
                                            C@address,
                                            8L)
               }
           },
           stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(C)
}

# vclMatrix colSums
vclMatrix_colSums <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
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
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(sums)
}

# vclMatrix rowSums
vclMatrix_rowSums <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
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
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(sums)
}

# vclMatrix colMeans
vclMatrix_colMeans <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
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
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(sums)
}

# vclMatrix rowMeans
vclMatrix_rowMeans <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
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
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(sums)
}

# GPU Pearson Covariance
vclMatrix_pmcc <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(A)
    
    B <- vclMatrix(nrow = ncol(A), ncol = ncol(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vclMatrix_pmcc(A@address, 
                                        B@address, 
                                        6L),
           "double" = cpp_vclMatrix_pmcc(A@address, 
                                         B@address,
                                         8L),
           stop("unsupported matrix type")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(B)
}

# GPU Euclidean Distance
vclMatrix_euclidean <- function(A, D, diag, upper, p, squareDist){
    
    assert_are_identical(A@.context_index, D@.context_index)
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(D)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vclMatrix_eucl(A@address, 
                                        D@address, 
                                        squareDist, 
                                        6L),
           "double" = cpp_vclMatrix_eucl(A@address, 
                                         D@address,
                                         squareDist,
                                         8L),
           stop("Unsupported matrix type")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    invisible(D)
}

# GPU Pairwise Euclidean Distance
vclMatrix_peuclidean <- function(A, B, D, squareDist){
    
    if(length(unique(c(A@.context_index, B@.context_index, D@.context_index))) != 1){
        stop("Context indices don't match between arguments")
    }
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(D)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vclMatrix_peucl(A@address,
                                         B@address,
                                        D@address, 
                                        squareDist, 
                                        6L),
           "double" = cpp_vclMatrix_peucl(A@address, 
                                          B@address,
                                         D@address,
                                         squareDist,
                                         8L),
           stop("Unsupported matrix type")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    invisible(D)
}

# GPU Element-Wise Absolute Value
vclMatElemAbs <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vclMatrix_elem_abs(A@address,
                                           C@address,
                                           6L)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vclMatrix_elem_abs(A@address,
                                            C@address,
                                            8L)
               }
           },
           stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(C)
}

# GPU Vector maximum
vclMatMax <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(A)
    
    C <- switch(type,
                integer = {
                    stop("integer not currently implemented")
                },
                float = {cpp_vclMatrix_max(A@address,
                                           6L)
                },
                double = {
                    if(!deviceHasDouble()){
                        stop("Selected GPU does not support double precision")
                    }else{cpp_vclMatrix_max(A@address,
                                            8L)
                    }
                },
                stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(C)
}

# GPU Vector minimum
vclMatMin <- function(A){
    
    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }
    
    type <- typeof(A)
    
    C <- switch(type,
                integer = {
                    stop("integer not currently implemented")
                },
                float = {cpp_vclMatrix_min(A@address,
                                           6L)
                },
                double = {
                    if(!deviceHasDouble()){
                        stop("Selected GPU does not support double precision")
                    }else{cpp_vclMatrix_min(A@address,
                                            8L)
                    }
                },
                stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(C)
}

# GPU Matrix transpose
vclMatrix_t <- function(A){
    
    type <- typeof(A)

    oldContext <- currentContext()
    if(oldContext != A@.context_index){
        setContext(A@.context_index)
    }

    B <- vclMatrix(0, ncol = nrow(A), nrow = ncol(A), type = type)

    switch(type,
           integer = {cpp_vclMatrix_transpose(A@address, B@address, 4L)},
           float = {cpp_vclMatrix_transpose(A@address, B@address, 6L)},
           double = {cpp_vclMatrix_transpose(A@address, B@address, 8L)},
           stop("type not recognized")
    )
    
    if(oldContext != A@.context_index){
        setContext(oldContext)
    }
    
    return(B)
}


