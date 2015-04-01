

#' @useDynLib bigGPU
#' @importFrom Rcpp evalCpp

#' @title GPU Vector Addition
#' @description vector addition
#' @export
gpu_vec_add <- function(A, B){
    
#     print(find.package("bigGPU", .libPaths()))
    
    pkg_path <- find.package("bigGPU", .libPaths())
    file <- file.path(pkg_path, "CL", "vector_add_kernel.cl")
#     print(file)
    
    if(!file_test("-f", file)){
        stop("kernel file does not exist")
    }
    C <- vector(mode = "integer", length=length(A))
    
    kernel <- readChar(file, file.info(file)$size)
    
    out <- as.gpuVector(cpp_gpu_vec_add(A,B,C,kernel))
    return(out)
}
