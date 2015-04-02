

#' @useDynLib gpuR
#' @importFrom Rcpp evalCpp

#' @title GPU Vector Addition
#' @description vector addition
# ' @export
gpu_vec_add <- function(A, B){
    
#     print('called addition')
    
    pkg_path <- find.package("gpuR", .libPaths())
    file <- file.path(pkg_path, "CL", "vector_add_kernel.cl")
#     print(file)
    
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
    
#     print("called subtraction function")
    
    pkg_path <- find.package("gpuR", .libPaths())
    file <- file.path(pkg_path, "CL", "vector_subtr_kernel.cl")
    #     print(file)
    
    if(!file_test("-f", file)){
        stop("kernel file does not exist")
    }
    C <- vector(mode = "integer", length=length(A))
    
    kernel <- readChar(file, file.info(file)$size)
    
    out <- as.gpuVector(cpp_gpu_two_vec(A,B,C,kernel, "vector_subtr"))
    return(out)
}