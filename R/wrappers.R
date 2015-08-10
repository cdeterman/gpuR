

#' @useDynLib gpuR
#' @importFrom Rcpp evalCpp
#' @import assertive


detectCPUs <- function(platform_idx=1L){
    assert_is_integer(platform_idx)
    assert_all_are_positive(platform_idx)
    
    out <- cpp_detectCPUs(platform_idx)
    return(out)
}

#' @title Detect Available GPUs
#' @description Find out how many GPUs available
#' @param platform_idx An integer value indicating which platform to query.
#' @return An integer representing the number of available GPUs
#' @seealso \link{detectPlatforms}
#' @export
detectGPUs <- function(platform_idx=1L){
    assert_is_integer(platform_idx)
    assert_all_are_positive(platform_idx)
    
    out <- cpp_detectGPUs(platform_idx)
    return(out)
}

#' @title GPU Information
#' @description Get basic information about selected GPU
#' @param platform_idx An integer value indicating which platform to query.
#' @param gpu_idx An integer value indicating which gpu to query.
#' @return \item{deviceName}{GPU Name}
#' @return \item{deviceVendor}{GPU Vendor}
#' @return \item{numberOfCores}{Number of Computing Units 
#'  (which execute the work groups)}
#' @return \item{maxWorkGroupSize}{Maximum number of work items
#'  per group}
#' @return \item{maxWorkItemDim}{Number of dimensions}
#' @return \item{maxWorkItemSizes}{Maximum number of works items
#'  per dimension}
#' @return \item{deviceMemory}{Global amount of memory (bytes)}
#' @return \item{clockFreq}{Maximum configured clock frequency of the 
#' device in MHz}
#' @return \item{localMem}{Maximum amount of local memory for each work 
#' group (bytes)}
#' @return \item{maxAllocatableMem}{Maximum amount of memory in a single 
#' piece (bytes)}
#' @return \item{available}{Whether the device is available}
#' @seealso \link{detectPlatforms} \link{detectGPUs}
#' @author Charles Determan Jr.
#' @export
gpuInfo <- function(platform_idx=1L, gpu_idx=1L){
    assert_is_integer(platform_idx)
    assert_all_are_positive(platform_idx)
    assert_is_integer(gpu_idx)
    assert_all_are_positive(gpu_idx)
    
    out <- cpp_gpuInfo(platform_idx, gpu_idx)
    return(out)
}


#' @title OpenCL Platform Information
#' @description Get basic information about the OpenCL platform
#' @param platform_idx An integer value to specify which platform to check
#' @author Charles Determan Jr.
#' @return \item{platformName}{Platform Name}
#' @return \item{platformVendor}{Platform Vendor}
#' @return \item{platformVersion}{Platform OpenCL Version}
#' @return \item{platformExtensions}{Avaiable platform extensions}
#' @export
platformInfo <- function(platform_idx=1L){
    assert_is_integer(platform_idx)
    assert_all_are_positive(platform_idx)
    
    out <- cpp_platformInfo(platform_idx)
    return(out)
}

#' @title Check GPU double precision support
#' @description This function checks the GPU device extensions for the
#' variable cl_khr_fp64 which means the device supports double precision.
#' @param platform_idx An integer value indicating which platform to query.
#' @param gpu_idx An integer value indicating which gpu to query.
#' @return A boolean designating whether the device supports double precision
#' @seealso \link{gpuInfo}
#' @export
deviceHasDouble <- function(platform_idx=1L, gpu_idx=1L){
    assert_is_integer(platform_idx)
    assert_all_are_positive(platform_idx)
    assert_is_integer(gpu_idx)
    assert_all_are_positive(gpu_idx)
    
    if(options("gpuR.default.device") == "cpu"){
        return(TRUE)
    }else{
        out <- cpp_device_has_double(platform_idx, gpu_idx)
        return(out)
    }
    
}


# GPU Vector Addition
gpu_vec_add <- function(A, B){
    
    pkg_path <- find.package("gpuR", .libPaths())
    file <- file.path(pkg_path, "CL", "vector_add_kernel.cl")
    
    if(!file_test("-f", file)){
        stop("kernel file does not exist")
    }

    C <- gpuVector(length = length(A), type="integer")
    
    kernel <- readChar(file, file.info(file)$size)
    
    cpp_gpu_two_vec(A@address,B@address,C@address,
                    kernel, "vector_add")
    return(C)
}


# GPU Vector Subtraction
gpu_vec_subtr <- function(A, B){
    
    pkg_path <- find.package("gpuR", .libPaths())
    file <- file.path(pkg_path, "CL", "vector_subtr_kernel.cl")
    
    if(!file_test("-f", file)){
        stop("kernel file does not exist")
    }
    
    C <- gpuVector(length = length(A), type="integer")
    
    kernel <- readChar(file, file.info(file)$size)
    
    cpp_gpu_two_vec(A@address,B@address,C@address,
                    kernel, "vector_subtr")
    return(C)
}


# GPU Matrix Multiplication
gpu_Mat_mult <- function(A, B){
    
    pkg_path <- find.package("gpuR", .libPaths())
    file <- file.path(pkg_path, "CL", "basic_gemm.cl")
    
    if(!file_test("-f", file)){
        stop("kernel file does not exist")
    }
    kernel <- readChar(file, file.info(file)$size)
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
                    )
               )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(B), type=type)
# 
#     integer = {cpp_gpuMatrix_igemm(A@address,
#                                    B@address, 
#                                    C@address,
#                                    kernel)
    switch(type,
                 integer = {
                     cpp_gpuMatrix_igemm(A@address,
                                         B@address, 
                                         C@address,
                                         kernel)
#                      cpp_vienna_gpuMatrix_igemm(A@address,
#                                                        B@address,
#                                                        C@address)
                  },
                  float = {cpp_vienna_gpuMatrix_sgemm(A@address,
                                                      B@address,
                                                      C@address,
                                                      device_flag)
                  },
                  double = {
                      if(!deviceHasDouble()){
                          stop("Selected GPU does not support double precision")
                      }else{cpp_vienna_gpuMatrix_dgemm(A@address,
                                                       B@address,
                                                       C@address,
                                                       device_flag)
                      }
                  },
                  {
                      stop("type not recognized")
                  })
#     rm(C)
    return(C)
}

# GPU Element-Wise Multiplication
gpuMatElemMult <- function(A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    if(!all(dim(A) == dim(B))){
        stop("matrices not conformable")
    }
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vienna_sgpuMatrix_elem_prod(A@address,
                                               B@address,
                                               C@address,
                                               device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuMatrix_elem_prod(A@address,
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

# GPU Element-Wise Division
gpuMatElemDiv <- function(A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    if(!all(dim(A) == dim(B))){
        stop("matrices not conformable")
    }
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vienna_sgpuMatrix_elem_div(A@address,
                                                    B@address,
                                                    C@address,
                                                   device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuMatrix_elem_div(A@address,
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
gpuMatElemSin <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vienna_sgpuMatrix_elem_sin(A@address,
                                                   C@address,
                                                   device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuMatrix_elem_sin(A@address,
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
gpuMatElemArcSin <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vienna_sgpuMatrix_elem_asin(A@address,
                                                   C@address,
                                                   device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuMatrix_elem_asin(A@address,
                                                    C@address,
                                                    device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Hyperbolic Sine
gpuMatElemHypSin <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vienna_sgpuMatrix_elem_sinh(A@address,
                                                   C@address,
                                                   device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuMatrix_elem_sinh(A@address,
                                                    C@address,
                                                    device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Cos
gpuMatElemCos <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vienna_sgpuMatrix_elem_cos(A@address,
                                                   C@address,
                                                   device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuMatrix_elem_cos(A@address,
                                                    C@address,
                                                    device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Arc Cos
gpuMatElemArcCos <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vienna_sgpuMatrix_elem_acos(A@address,
                                                   C@address,
                                                   device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuMatrix_elem_acos(A@address,
                                                    C@address,
                                                    device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Hyperbolic Cos
gpuMatElemHypCos <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vienna_sgpuMatrix_elem_cosh(A@address,
                                                   C@address,
                                                   device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuMatrix_elem_cosh(A@address,
                                                    C@address,
                                                    device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Tan
gpuMatElemTan <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1L, 
               "gpu" = 0L,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vienna_sgpuMatrix_elem_tan(A@address,
                                                   C@address,
                                                   device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuMatrix_elem_tan(A@address,
                                                    C@address,
                                                    device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Arc Tan
gpuMatElemArcTan <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vienna_sgpuMatrix_elem_atan(A@address,
                                                   C@address,
                                                   device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuMatrix_elem_atan(A@address,
                                                    C@address,
                                                    device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU Element-Wise Hyperbolic Tan
gpuMatElemHypTan <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(A), type=type)
    
    switch(type,
           integer = {
               stop("integer not currently implemented")
           },
           float = {cpp_vienna_sgpuMatrix_elem_tanh(A@address,
                                                   C@address,
                                                   device_flag)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_dgpuMatrix_elem_tanh(A@address,
                                                    C@address,
                                                    device_flag)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# GPU axpy wrapper
gpu_Mat_axpy <- function(alpha, A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    nrA = nrow(A)
    ncA = ncol(A)
    nrB = nrow(B)
    ncB = ncol(B)
    
    pkg_path <- find.package("gpuR", .libPaths())
    file <- file.path(pkg_path, "CL", "basic_axpy.cl")
    
    if(!file_test("-f", file)){
        stop("kernel file does not exist")
    }
    kernel <- readChar(file, file.info(file)$size)
    
    type <- typeof(A)
    
    Z <- gpuMatrix(nrow=nrB, ncol=ncA, type=type)
    if(!missing(B))
    {
        if(length(B[]) != length(A[])) stop("Lengths of matrices must match")
        Z@address <- B@address
    }
    
    switch(type,
           integer = {cpp_gpuMatrix_iaxpy(alpha, 
                                          A@address,
                                          Z@address, 
                                          kernel)
           },
           float = {cpp_vienna_gpuMatrix_saxpy(alpha, 
                                               A@address, 
                                               Z@address, 
                                               device_flag)
           },
           double = {cpp_vienna_gpuMatrix_daxpy(alpha, 
                                                A@address,
                                                Z@address,
                                                device_flag)
           },
        {
            stop("type not recognized")
        }
    )

    return(Z)
}

# GPU colSums
gpu_colSums <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    if(type == "integer"){
        stop("integer type not currently implemented")
    }
    
    sums <- gpuVector(length = ncol(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vienna_fgpuMatrix_colsum(A@address, sums@address, device_flag),
           "double" = cpp_vienna_dgpuMatrix_colsum(A@address, sums@address, device_flag)
           )
    
    return(sums)
}

# GPU rowSums
gpu_rowSums <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    if(type == "integer"){
        stop("integer type not currently implemented")
    }
    
    sums <- gpuVector(length = nrow(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vienna_fgpuMatrix_rowsum(A@address, sums@address, device_flag),
           "double" = cpp_vienna_dgpuMatrix_rowsum(A@address, sums@address, device_flag)
    )
    
    return(sums)
}

# GPU colMeans
gpu_colMeans <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    if(type == "integer"){
        stop("integer type not currently implemented")
    }
    
    sums <- gpuVector(length = ncol(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vienna_fgpuMatrix_colmean(A@address, sums@address, device_flag),
           "double" = cpp_vienna_dgpuMatrix_colmean(A@address, sums@address, device_flag)
    )
    
    return(sums)
}

# GPU rowMeans
gpu_rowMeans <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    if(type == "integer"){
        stop("integer type not currently implemented")
    }
    
    sums <- gpuVector(length = nrow(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vienna_fgpuMatrix_rowmean(A@address, sums@address, device_flag),
           "double" = cpp_vienna_dgpuMatrix_rowmean(A@address, sums@address, device_flag)
    )
    
    return(sums)
}

# GPU Pearson Covariance
gpu_pmcc <- function(A){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(A)
    
    B <- gpuMatrix(nrow = ncol(A), ncol = ncol(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vienna_fgpuMatrix_pmcc(A@address, B@address, device_flag),
           "double" = cpp_vienna_dgpuMatrix_pmcc(A@address, B@address, device_flag)
    )
    
    return(B)
}

# GPU crossprod
gpu_crossprod <- function(X, Y){
    
    if(ncol(X) != ncol(Y)){
        stop("matrices non-conformable")
    }
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(X)
    
    Z <- gpuMatrix(nrow = ncol(X), ncol = ncol(Y), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vienna_gpuMatrix_scrossprod(X@address, Y@address, Z@address, device_flag),
           "double" = cpp_vienna_gpuMatrix_dcrossprod(X@address, Y@address, Z@address, device_flag)
    )
    
    return(Z)
}

# GPU tcrossprod
gpu_tcrossprod <- function(X, Y){
    
    if(nrow(X) != nrow(Y)){
        stop("matrices non-conformable")
    }
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    type <- typeof(X)
    
    Z <- gpuMatrix(nrow = nrow(X), ncol = nrow(Y), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vienna_gpuMatrix_stcrossprod(X@address, Y@address, Z@address, device_flag),
           "double" = cpp_vienna_gpuMatrix_dtcrossprod(X@address, Y@address, Z@address, device_flag)
    )
    
    return(Z)
}
