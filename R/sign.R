
# GPU Element-Wise sign
gpuMatSign <- function(A){
    
    type <- typeof(A)
    
    if(is(A, "vclMatrix")){
        B <- vclMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)    
    }else{
        B <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type, ctx_id = A@.context_index)
    }
    
    
    maxWorkGroupSize <- 
        switch(deviceType(B@.platform_index, B@.device_index),
               "gpu" = gpuInfo(B@.platform_index, B@.device_index)$maxWorkGroupSize,
               "cpu" = cpuInfo(B@.platform_index, B@.device_index)$maxWorkGroupSize,
               stop("unrecognized device type")
        )
    
    
    switch(type,
           integer = {
               file <- system.file("CL", "iMatSign.cl", package = "gpuR")
               
               if(!file_test("-f", file)){
                   stop("kernel file does not exist")
               }
               kernel <- readChar(file, file.info(file)$size)
               
               cpp_vclMatrix_sign(A@address,
                                  is(A, "vclMatrix"),
                                  B@address,
                                  is(B, "vclMatrix"),
                                  kernel,
                                  sqrt(maxWorkGroupSize),
                                  4L,
                                  A@.context_index - 1L)
           },
           float = {
               file <- system.file("CL", "fMatSign.cl", package = "gpuR")
               
               if(!file_test("-f", file)){
                   stop("kernel file does not exist")
               }
               kernel <- readChar(file, file.info(file)$size)
               
               cpp_vclMatrix_sign(A@address,
                                  is(A, "vclMatrix"),
                                  B@address,
                                  is(B, "vclMatrix"),
                                  kernel,
                                  sqrt(maxWorkGroupSize),
                                  6L,
                                  A@.context_index - 1L)
           },
           double = {
               file <- system.file("CL", "dMatSign.cl", package = "gpuR")
               
               if(!file_test("-f", file)){
                   stop("kernel file does not exist")
               }
               kernel <- readChar(file, file.info(file)$size)
               
               cpp_vclMatrix_sign(A@address,
                                  is(A, "vclMatrix"),
                                  B@address,
                                  is(B, "vclMatrix"),
                                  kernel,
                                  sqrt(maxWorkGroupSize),
                                  8L,
                                  A@.context_index - 1L)
           },
           stop("type not recognized")
    )
    
    return(B)
}


# GPU Element-Wise sign
gpuVecSign <- function(A){
    
    type <- typeof(A)
    
    if(is(A, "vclVector")){
        B <- vclVector(length = length(A), type=type, ctx_id = A@.context_index)    
    }else{
        B <- gpuVector(length = length(A), type=type, ctx_id = A@.context_index)
    }
    
    
    maxWorkGroupSize <- 
        switch(deviceType(B@.platform_index, B@.device_index),
               "gpu" = gpuInfo(B@.platform_index, B@.device_index)$maxWorkGroupSize,
               "cpu" = cpuInfo(B@.platform_index, B@.device_index)$maxWorkGroupSize,
               stop("unrecognized device type")
        )
    
    
    switch(type,
           integer = {
               file <- system.file("CL", "iVecSign.cl", package = "gpuR")
               
               if(!file_test("-f", file)){
                   stop("kernel file does not exist")
               }
               kernel <- readChar(file, file.info(file)$size)
               
               cpp_vclVector_sign(A@address,
                                  is(A, "vclVector"),
                                  B@address,
                                  is(B, "vclVector"),
                                  kernel,
                                  maxWorkGroupSize,
                                  4L,
                                  A@.context_index - 1L)
           },
           float = {
               file <- system.file("CL", "fVecSign.cl", package = "gpuR")
               
               if(!file_test("-f", file)){
                   stop("kernel file does not exist")
               }
               kernel <- readChar(file, file.info(file)$size)
               
               cpp_vclVector_sign(A@address,
                                  is(A, "vclVector"),
                                  B@address,
                                  is(B, "vclVector"),
                                  kernel,
                                  maxWorkGroupSize,
                                  6L,
                                  A@.context_index - 1L)
           },
           double = {
               file <- system.file("CL", "dVecSign.cl", package = "gpuR")
               
               if(!file_test("-f", file)){
                   stop("kernel file does not exist")
               }
               kernel <- readChar(file, file.info(file)$size)
               
               cpp_vclVector_sign(A@address,
                                  is(A, "vclVector"),
                                  B@address,
                                  is(B, "vclVector"),
                                  kernel,
                                  maxWorkGroupSize,
                                  8L,
                                  A@.context_index - 1L)
           },
           stop("type not recognized")
    )
    
    return(B)
}

