

#' @useDynLib gpuR
#' @importFrom Rcpp evalCpp
#' @import assertive

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


#' @title GPU Vector Addition
#' @description vector addition
#' @param A A gpuVector object
#' @param B A gpuVector object
#' @return A gpuVector object
#' @author Charles Determan Jr.
gpu_vec_add <- function(A, B){
    
    pkg_path <- find.package("gpuR", .libPaths())
    file <- file.path(pkg_path, "CL", "vector_add_kernel.cl")
    
    if(!file_test("-f", file)){
        stop("kernel file does not exist")
    }
    C <- vector(mode = "integer", length=length(A))
    
    kernel <- readChar(file, file.info(file)$size)
    
    out <- as.gpuVector(cpp_gpu_two_vec(A,B,C,kernel, "vector_add"))
    return(out)
}


#' @title GPU Vector Subtraction
#' @description vector subtraction
#' @param A A gpuVector object
#' @param B A gpuVector object
#' @return A gpuVector object
#' @author Charles Determan Jr.
gpu_vec_subtr <- function(A, B){
    
    pkg_path <- find.package("gpuR", .libPaths())
    file <- file.path(pkg_path, "CL", "vector_subtr_kernel.cl")
    
    if(!file_test("-f", file)){
        stop("kernel file does not exist")
    }
    C <- vector(mode = "integer", length=length(A))
    
    kernel <- readChar(file, file.info(file)$size)
    
    out <- as.gpuVector(cpp_gpu_two_vec(A,B,C,kernel, "vector_subtr"))
    return(out)
}


#' @title GPU Big Matrix Multiplication
#' @description big matrix gpu multiplication
#' @param A A gpuBigMatrix object
#' @param B A gpuBigMatrix object
#' @return A gpuBigMatrix object
#' @import bigalgebra 
#' @import bigmemory
#' @author Charles Determan Jr.
gpu_BigMat_mult <- function(A, B){
    
    nrA = nrow(A)
    ncA = ncol(A)
    nrB = nrow(B)
    ncB = ncol(B)
    
    pkg_path <- find.package("gpuR", .libPaths())
#     file <- file.path(pkg_path, "CL", "basic_matrix_mult_kernel.cl")
    file <- file.path(pkg_path, "CL", "basic_gemm.cl")
    
    if(!file_test("-f", file)){
        stop("kernel file does not exist")
    }
    kernel <- readChar(file, file.info(file)$size)
    
    type <- typeof(A)
#     C <- bigalgebra:::anon_matrix(nrB, ncA, type=type)


    C <- gpuBigMatrix(big.matrix(nrB, ncA, type=type), type=type)

    switch(typeof(C),
           integer = {cpp_gpuBigMatrix_igemm(A,B,C, kernel, "iMatMult")
           },
           float = {cpp_vienna_gpuBigMatrix_sgemm(A,B,C)
           },
           double = {cpp_vienna_gpuBigMatrix_dgemm(A,B,C)
           },
           {
               stop("matrix type not defined")
           })

    return(C)
}


#' @title GPU Matrix Multiplication
#' @description matrix multiplication
#' @param A A gpuMatrix object
#' @param B A gpuMatrix object
#' @return A gpuMatrix object
#' @import bigalgebra 
#' @import bigmemory
#' @author Charles Determan Jr.
gpu_Mat_mult <- function(A, B){
    
    pkg_path <- find.package("gpuR", .libPaths())
    file <- file.path(pkg_path, "CL", "basic_gemm.cl")
    
    if(!file_test("-f", file)){
        stop("kernel file does not exist")
    }
    kernel <- readChar(file, file.info(file)$size)
    
    type <- typeof(A)

    out <- switch(type,
                  integer = {
                      new("igpuMatrix", 
                          x=cpp_gpuMatrix_igemm(A@x,B@x, kernel, "iMatMult"),
                          type="integer"
                      )
                  },
                  float = {
                      new("fgpuMatrix", 
                          x=cpp_vienna_gpuMatrix_sgemm(A@x,B@x),
                          type="float"
                      )
                  },
                  double = {
                      if(!deviceHasDouble()){
                          stop("Selected GPU does not support double precision")
                      }else{
                          new("dgpuMatrix", 
                              x=cpp_vienna_gpuMatrix_dgemm(A@x,B@x),
                              type="double"
                          )
                      }
                  },
                  {
                      stop("type not recognized")
                  })
#     rm(C)
    return(out)
}


#' @title GPU Matrix Multiplication
#' @description matrix multiplication
#' @param alpha Numeric value to multiply the A matrix
#' @param A A gpuMatrix object
#' @param B A gpuMatrix object
#' @return A gpuMatrix object
#' @import bigalgebra 
#' @import bigmemory
#' @author Charles Determan Jr.
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
    
    Z <- matrix(0, nrow=nrB, ncol=ncA)
    if(!missing(B))
    {
        if(length(B@x) != length(A@x)) stop("Lengths of matrices must match")
        Z <- B@x
    }
    
    out <- switch(type,
                  integer = {
                      new("igpuMatrix", 
                          x=cpp_gpuMatrix_iaxpy(alpha, A@x,Z, kernel, "iaxpy"),
                          type="integer"
                      )
                  },
                  float = {
                      new("fgpuMatrix", 
                          x=cpp_vienna_gpuMatrix_saxpy(alpha, A@x, Z),
                          type="float"
                      )
                  },
                  double = {
                      new("dgpuMatrix", 
                          x=cpp_vienna_gpuMatrix_daxpy(alpha, A@x,Z),
                          type="double"
                      )
                  },
                  {
                      stop("type not recognized")
                  }
            )

    return(out)
}


#' @title GPU Big Matrix (X)AXPY
#' @description big matrix gpu daxpy
#' @param alpha Numeric value to multiple the A matrix
#' @param A A gpuBigMatrix object
#' @param B A gpuBigMatrix object
#' @return A gpuBigMatrix object
#' @import bigalgebra 
#' @import bigmemory
#' @author Charles Determan Jr.
gpu_BigMat_axpy <- function(alpha, A, B){
    
    nrA = nrow(A)
    ncA = ncol(A)
    nrB = nrow(B)
    ncB = ncol(B)
    
    if(abs(alpha) != 1){
        stop("alpha can only be 1 or -1")
    }
    
    pkg_path <- find.package("gpuR", .libPaths())
    #     file <- file.path(pkg_path, "CL", "basic_matrix_mult_kernel.cl")
    file <- file.path(pkg_path, "CL", "basic_axpy.cl")
    
    if(!file_test("-f", file)){
        stop("kernel file does not exist")
    }
    kernel <- readChar(file, file.info(file)$size)
    
    type <- typeof(A)    
    
#     options(bigmemory.typecast.warning=FALSE)
    
    Z <- gpuBigMatrix(big.matrix(nrB, ncA, type=type), type=type)
    if(!missing(B))
    {
#         options(bigmemory.typecast.warning=FALSE)
        if(length(B) != length(A)) stop("Lengths of matrices must match")
        Z[] <- B[]
    }
    
    switch(typeof(Z),
           integer = {
               cpp_gpuBigMatrix_iaxpy(alpha, A, Z, kernel, "iaxpy")
           },
           float = {
               cpp_vienna_gpuBigMatrix_saxpy(alpha, A, Z)
           },
           double = {
               cpp_vienna_gpuBigMatrix_daxpy(alpha, A, Z)
           },
           {
               stop("matrix type not defined")
           }
    )

    return(Z)
}

# #' @export
# test_tmp_matrix <- function(A, B){
#     
#     ncA = ncol(A)
#     nrB = nrow(B)
#     
#     type <- typeof(A)
#     C <- bigalgebra:::anon_matrix(nrB, ncA, type=type)
# 
#     out <- new("igpuMatrix", address=C@address)
#     return(out)
# }


