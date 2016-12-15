
#' @export
vclMat_logistic <- function(A){
    
    type <- typeof(A)
    ctx_id <- A@.context_index - 1
    
    switch(type,
           "float" = cpp_vclMatrix_logistic(A@address, ctx_id, 6L),
           "double" = cpp_vclMatrix_logistic(A@address, ctx_id, 8L),
           stop("unimplemented type")
    )
}

# vclMatrix log deriv

#' @export
vclMat_log_deriv <- function(A, B, inplace = FALSE){
    
    if(inplace){
        Z <- A
    }else{
        Z <- deepcopy(A)
    }
    
    type <- typeof(Z)
    
    maxWorkGroupSize <- 
        switch(deviceType(Z@.platform_index, Z@.device_index),
               "gpu" = gpuInfo(Z@.platform_index, Z@.device_index)$maxWorkGroupSize,
               "cpu" = cpuInfo(Z@.platform_index, Z@.device_index)$maxWorkGroupSize,
               stop("unrecognized device type")
        )
    
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



#' @export
unlist_gpuRvec <- function(x){
	
	tmp <- x[[1]]
	type <- typeof(tmp)
	ctx_id <- tmp@.context_index - 1
	
	len <- Reduce(`+`, lapply(x, length))
	
	vec <- vclVector(length = len, type = type, ctx_id = tmp@.context_index)
	
	start = 1L
	end = 0L
	
	for(elem in x){
		end = end + length(elem)
		
		# print(start)
		# print(end)
		vec[start:end] <- elem
		start <- start + end
		
	}
	
	return(vec)
}


#' @export
gpuR_rprop_plus <- function(gradients, gradients.old, weights_in, nrow.weights, ncol.weights,
                            learningrate, learningrate.factor, learningrate.limit, exclude)
{
    # weights <- unlist(weights)
    weights <- unlist_gpuR(weights_in)
    
    maxWorkGroupSize <- 
        switch(deviceType(gradients@.platform_index, gradients@.device_index),
               "gpu" = gpuInfo(gradients@.platform_index, gradients@.device_index)$maxWorkGroupSize,
               "cpu" = cpuInfo(gradients@.platform_index, gradients@.device_index)$maxWorkGroupSize,
               stop("unrecognized device type")
        )
    
    type <- switch(typeof(gradients),
                   "float" = 6L,
                   "double" = 8L,
                   stop("unimplemented type"))
    
    kernel <- switch(typeof(gradients),
                     "float" = {
                         file <- system.file("CL", "frprop_plus.cl", package = "gpuR")
                         
                         if(!file_test("-f", file)){
                             stop("kernel file does not exist")
                         }
                         
                         readChar(file, file.info(file)$size)
                     },
                     "double" = {
                         file <- system.file("CL", "drprop_plus.cl", package = "gpuR")
                         
                         if(!file_test("-f", file)){
                             stop("kernel file does not exist")
                         }
                         
                         readChar(file, file.info(file)$size)
                     })
    
    gpuR:::cpp_gpu_rprop_plus(gradients@address, gradients.old@address, weights@address, 
                              learningrate@address, learningrate.factor, learningrate.limit, 
                              kernel, sqrt(maxWorkGroupSize), gradients@.context_index - 1, type)
    
}