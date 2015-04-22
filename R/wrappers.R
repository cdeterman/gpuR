

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
#' @export
platformInfo <- function(platform_idx=1L){
    assert_is_integer(platform_idx)
    assert_all_are_positive(platform_idx)
    
    out <- cpp_platformInfo(platform_idx)
    return(out)
}


#' @title GPU Vector Addition
#' @description vector addition
# ' @export
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
# ' @export
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


#' @title GPU Matrix Multiplication
#' @description matrix multiplication
#' @import bigalgebra 
#' @import bigmemory
# ' @export
gpu_mat_mult <- function(A, B){
    
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
    C <- bigalgebra:::anon_matrix(nrB, ncA, type=type)
    
#     newMat <- as.integer(cpp_gpu_two_mat2(A,B,C,kernel, "iMatMult"))
#     newMat <- as.integer(cpp_gpu_sgemm(A@address,B@address,C@address))

    out <- switch(typeof(C),
                  integer = {
                      new("igpuMatrix", 
                          address=cpp_gpu_mat_mult(A@address,B@address,C@address, kernel, "iMatMult")
                      )
                  },
                  float = {
                      new("fgpuMatrix", 
                          address=cpp_gpu_sgemm(A@address,B@address,C@address)
                      )
                  },
                  double = {
                      new("dgpuMatrix", 
                          address=cpp_gpu_dgemm(A@address,B@address,C@address)
                      )
                  })
#     out <- new("igpuMatrix", 
#                address=cpp_gpu_sgemm(A@address,B@address,C@address)
#     )

#     print("New Size")
#     print(length(newMat))
#     print(newMat)
#     print(str(newMat))

#     out <- gpuMatrix(newMat, 
#                      nrow=nrA, ncol=ncB,
#                      byrow=TRUE)
    # cleanup
#     gc()

    return(out)
}

#' @export
test_tmp_matrix <- function(A, B){
    
    ncA = ncol(A)
    nrB = nrow(B)
    
    type <- typeof(A)
    C <- bigalgebra:::anon_matrix(nrB, ncA, type=type)

    out <- new("igpuMatrix", address=C@address)
    return(out)
}


