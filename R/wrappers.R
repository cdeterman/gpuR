

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
    
    out <- cpp_device_has_double(platform_idx, gpu_idx)
    return(out)
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
    
    type <- typeof(A)
    
    C <- gpuMatrix(nrow=nrow(A), ncol=ncol(B), type=type)

    switch(type,
                  integer = {cpp_gpuMatrix_igemm(A@address,
                                                      B@address, 
                                                      C@address,
                                                      kernel)
                  },
                  float = {cpp_vienna_gpuMatrix_sgemm(A@address,
                                                             B@address,
                                                             C@address)
                  },
                  double = {
                      if(!deviceHasDouble()){
                          stop("Selected GPU does not support double precision")
                      }else{cpp_vienna_gpuMatrix_dgemm(A@address,
                                                                 B@address,
                                                                 C@address)
                      }
                  },
                  {
                      stop("type not recognized")
                  })
#     rm(C)
    return(C)
}


# GPU axpy wrapper
gpu_Mat_axpy <- function(alpha, A, B){
    
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
                                               Z@address)
           },
           double = {cpp_vienna_gpuMatrix_daxpy(alpha, 
                                                A@address,
                                                Z@address)
           },
        {
            stop("type not recognized")
        }
    )

    return(Z)
}


gpu_colSums <- function(A){
    
    type <- typeof(A)
    
    if(type == "integer"){
        stop("integer type not currently implemented")
    }
    
    sums <- gpuVector(length = ncol(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vienna_fgpuMatrix_colsum(A@address, sums@address),
           "double" = cpp_vienna_dgpuMatrix_colsum(A@address, sums@address)
           )
    
    return(sums)
}

gpu_rowSums <- function(A){
    
    type <- typeof(A)
    
    if(type == "integer"){
        stop("integer type not currently implemented")
    }
    
    sums <- gpuVector(length = nrow(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vienna_fgpuMatrix_rowsum(A@address, sums@address),
           "double" = cpp_vienna_dgpuMatrix_rowsum(A@address, sums@address)
    )
    
    return(sums)
}


gpu_colMeans <- function(A){
    
    type <- typeof(A)
    
    if(type == "integer"){
        stop("integer type not currently implemented")
    }
    
    sums <- gpuVector(length = ncol(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vienna_fgpuMatrix_colmean(A@address, sums@address),
           "double" = cpp_vienna_dgpuMatrix_colmean(A@address, sums@address)
    )
    
    return(sums)
}

gpu_rowMeans <- function(A){
    
    type <- typeof(A)
    
    if(type == "integer"){
        stop("integer type not currently implemented")
    }
    
    sums <- gpuVector(length = nrow(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vienna_fgpuMatrix_rowmean(A@address, sums@address),
           "double" = cpp_vienna_dgpuMatrix_rowmean(A@address, sums@address)
    )
    
    return(sums)
}


gpu_pmcc <- function(A){
    
    type <- typeof(A)
    
    B <- gpuMatrix(nrow = ncol(A), ncol = ncol(A), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vienna_fgpuMatrix_pmcc(A@address, B@address),
           "double" = cpp_vienna_dgpuMatrix_pmcc(A@address, B@address)
    )
    
    return(B)
}

# #' @title GPU Pearson Covariance Calculation
# #' @description Calculates the pairwise pearson covariance for a matrix
# #' @param A A gpuMatrix object
# #' @return A gpuMatrix object
# #' @author Charles Determan Jr.
# gpu_pmcc <- function(A){
#     
#     pkg_path <- find.package("gpuR", .libPaths())
#     file <- file.path(pkg_path, "CL", "basic_pmcc.cl")
#     
#     if(!file_test("-f", file)){
#         stop("kernel file does not exist")
#     }
#     kernel <- readChar(file, file.info(file)$size)
#     
#     type <- typeof(A)
#     
#     B <- gpuMatrix(nrow=ncol(A), ncol=ncol(A), type=type)
#     
#     switch(type,
#            integer = {
#                stop("Integer not currently supported")
#                cpp_gpuMatrix_ipmcc(A@address, B@address, kernel)
#            },
#            float = {cpp_gpuMatrix_spmcc(A@address, B@address, kernel)
#            },
#            double = {
#                stop("Double not currently supported")
#                if(!deviceHasDouble()){
#                    stop("Selected GPU does not support double precision")
#                }else{
#                    cpp_gpuMatrix_dpmcc(A@address, B@address, kernel)
#                }
#            },
# {
#     stop("type not recognized")
# })
# return(B)
# }
# 
# 
# #' @export
# gpu_colMeans <- function(A){
#     
#     pkg_path <- find.package("gpuR", .libPaths())
#     file <- file.path(pkg_path, "CL", "basic_colMeans.cl")
#     
#     if(!file_test("-f", file)){
#         stop("kernel file does not exist")
#     }
#     kernel <- readChar(file, file.info(file)$size)
#     
#     type <- typeof(A)
#     
#     switch(type,
#            integer = {
#                stop("Integer not currently supported")
#                cpp_gpuMatrix_icolMeans(A@address, kernel)
#            },
#            float = {cpp_gpuMatrix_scolMeans(A@address, kernel)
#            },
#            double = {
#                stop("Double not currently supported")
#                if(!deviceHasDouble()){
#                    stop("Selected GPU does not support double precision")
#                }else{
#                    cpp_gpuMatrix_dcolMeans(A@address, kernel)
#                }
#            },
# {
#     stop("type not recognized")
# })
# }
