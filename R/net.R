
# vclMatrix log deriv

#' @export
vclMat_log_deriv <- function(A, B, inplace = FALSE){
    
    if(inplace){
        Z <- A
    }else{
        Z <- deepcopy(A)
    }
    
    type <- typeof(Z)
    
    switch(type,
           integer = {
               stop("not currently implemented")
               # file <- system.file("CL", "iScalarAXPY.cl", package = "gpuR")
               # 
               # if(!file_test("-f", file)){
               #     stop("kernel file does not exist")
               # }
               # kernel <- readChar(file, file.info(file)$size)
               # 
               # cpp_vclMatrix_log_deriv(Z@address,
               #                           kernel,
               #                           Z@.context_index - 1,
               #                           4L)
           },
           float = {
               
               file <- system.file("CL", "fLogDeriv.cl", package = "gpuR")
               
               if(!file_test("-f", file)){
                   stop("kernel file does not exist")
               }
               kernel <- readChar(file, file.info(file)$size)
               
               cpp_vclMatrix_log_deriv(Z@address,
                                       B@address,
                                       sqrt(maxWorkGroupSize),
                                       kernel,
                                       Z@.context_index - 1,
                                       6L)
           },
           double = {
               
               file <- system.file("CL", "dLogDeriv.cl", package = "gpuR")
               
               if(!file_test("-f", file)){
                   stop("kernel file does not exist")
               }
               kernel <- readChar(file, file.info(file)$size)
               
               cpp_vclMatrix_log_deriv(Z@address,
                                       B@address,
                                       sqrt(maxWorkGroupSize),
                                       kernel,
                                       Z@.context_index - 1,
                                       8L)
           },
           stop("type not recognized")
    )
    
    if(inplace){
        return(invisible(Z))
    }else{
        return(Z)
    }
}


#' @export
max_abs <- function(x){
    return(vclVecElemMaxAbs(x))
}


#' @export
unlist_gpuR <- function(x){
    
    tmp <- x[[1]]
    type <- typeof(tmp)
    ctx_id <- tmp@.context_index - 1
    
    len <- Reduce(`+`, lapply(x, length))
    
    vec <- vclVector(length = len, type = type, ctx_id = tmp@.context_index)
    
    switch(type,
           integer = {
               vectorizeList(x, vec@address, ctx_id, 4L)
           },
           float = {
               vectorizeList(x, vec@address, ctx_id, 6L)
           },
           double = {
               vectorizeList(x, vec@address, ctx_id, 8L)
           },
           stop("unimplemented type")
    )
    
    return(invisible(vec))
}

