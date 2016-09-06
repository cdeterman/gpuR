#' @import assertive


#' @title Check device type
#' @description Check what type a device is given platform and device indices
#' @param platform_idx An integer value indicating which platform to query.
#' @param device_idx An integer value indicating which device to query.
#' @return A character string indicating the device type
#' @export
deviceType <- function(platform_idx = 1L, device_idx = 1L)
{
    assert_is_integer(platform_idx)
    assert_all_are_positive(platform_idx)
    assert_is_integer(device_idx)
    assert_all_are_positive(device_idx)
    
    out <- cpp_deviceType(platform_idx, device_idx)
    
    return(out)
}

#' @title Detect Available OpenCL enabled CPUs
#' @description Find out how many CPUs available
#' @param platform_idx An integer value indicating which platform to query.
#' If NULL it will iterate over all platforms and sum results
#' @return An integer representing the number of available CPUs
#' @seealso \link{detectPlatforms} \link{detectGPUs}
#' @export
detectCPUs <- function(platform_idx=NULL){
    
    current_context_id <- currentContext()
    
#     cpus <- try(cpp_detectCPUs(platform_idx), silent=TRUE)
#     if(class(cpus)[1] == "try-error"){
#         # need to make sure if errors out to switch back to original context
#         setContext(current_context_id)
#         return(0)
#     }else{
#         setContext(current_context_id)
#         return(cpus)
#     }
    
    if(is.null(platform_idx)){
        total_cpus = 0
        for(p in seq(detectPlatforms())){
            cpus <- try(cpp_detectCPUs(p), silent=TRUE)
            if(class(cpus)[1] == "try-error"){
                # need to make sure if errors out to switch back to original context
                total_cpus = total_cpus + 0
            }else{
                total_cpus = total_cpus + cpus
            }
        }
        setContext(current_context_id)
        
        return(total_cpus)
        
    }else{
        assert_is_integer(platform_idx)
        assert_all_are_positive(platform_idx)
        
        cpus <- try(cpp_detectCPUs(platform_idx), silent=TRUE)
        if(class(cpus)[1] == "try-error"){
            # need to make sure if errors out to switch back to original context
            setContext(current_context_id)
            return(0)
        }else{
            setContext(current_context_id)
            return(cpus)
        }
    }
}

#' @title Detect Available GPUs
#' @description Find out how many GPUs available
#' @param platform_idx An integer value indicating which platform to query.
#' If NULL it will iterate over all platforms and sum results
#' @return An integer representing the number of available GPUs
#' @seealso \link{detectPlatforms}
#' @export
detectGPUs <- function(platform_idx=NULL){
    
    current_context_id <- currentContext()
    
    if(is.null(platform_idx)){
        total_gpus = 0
        for(p in seq(detectPlatforms())){
            gpus <- try(cpp_detectGPUs(p), silent=TRUE)
            if(class(gpus)[1] == "try-error"){
                # need to make sure if errors out to switch back to original context
                total_gpus = total_gpus + 0
            }else{
                total_gpus = total_gpus + gpus
            }
        }
        setContext(current_context_id)
        
        return(total_gpus)
        
    }else{
        assert_is_integer(platform_idx)
        assert_all_are_positive(platform_idx)
        
        gpus <- try(cpp_detectGPUs(platform_idx), silent=TRUE)
        if(class(gpus)[1] == "try-error"){
            # need to make sure if errors out to switch back to original context
            setContext(current_context_id)
            return(0)
        }else{
            setContext(current_context_id)
            return(gpus)
        }
    }
    
}

#' @title Device Information
#' @description Get basic information about selected device (e.g. GPU)
#' @param platform_idx An integer value indicating which platform to query.
#' @param device_idx An integer value indicating which device to query.
#' @return \item{deviceName}{Device Name}
#' @return \item{deviceVendor}{Device Vendor}
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
#' @return \item{deviceExtensions}{OpenCL device extensions available}
#' @return \item{double_support}{Logical value if double type supported}
#' @seealso \link{detectPlatforms} \link{detectGPUs} \link{detectCPUs} \link{cpuInfo}
#' @author Charles Determan Jr.
#' @rdname deviceInfo
#' @aliases gpuInfo
#' @export
gpuInfo <- function(platform_idx=NULL, device_idx=NULL){
    
    if(detectGPUs(platform_idx) == 0){
        stop("No CPUs found on platform")
    }
    
    if(!is.null(platform_idx) && !is.null(device_idx)){
        assert_is_integer(platform_idx)
        assert_all_are_positive(platform_idx)
        assert_is_integer(device_idx)
        assert_all_are_positive(device_idx)
        
        if(device_idx > detectGPUs(platform_idx)){
            stop("Device index out of range on platform")
        }
        
    }else{
        contexts <- listContexts()
        idx <- which(contexts$device_type == 'gpu')
        if(length(idx) == 0){
            stop("No GPUs found in intialized contexts")
        }else{
            platform_idx <- contexts$platform_index[idx[1]] + 1L
            device_idx <- contexts$device_index[idx[1]] + 1L
        }
    }
    
    out <- cpp_gpuInfo(platform_idx, device_idx)
    return(out)
}

#' @rdname deviceInfo
#' @aliases cpuInfo
#' @export
cpuInfo <- function(platform_idx=NULL, device_idx=NULL){
    
    if(detectCPUs(platform_idx) == 0){
        stop("No CPUs found on platform")
    }
    
    if(!is.null(platform_idx) && !is.null(device_idx)){
        assert_is_integer(platform_idx)
        assert_all_are_positive(platform_idx)
        assert_is_integer(device_idx)
        assert_all_are_positive(device_idx)
        
        if(device_idx > detectCPUs(platform_idx)){
            stop("Device index out of range on platform")
        }
        
    }else{
        contexts <- listContexts()
        idx <- which(contexts$device_type == 'cpu')
        if(length(idx) == 0){
            stop("No CPUs found in intialized contexts")
        }else{
            platform_idx <- contexts$platform_index[idx[1]] + 1
            device_idx <- contexts$device_index[idx[1]] + 1
        }
    }
    
    out <- cpp_cpuInfo(platform_idx, device_idx)
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
    
    device_type <- currentDevice()$device_type
    
    out <- switch(device_type,
                  "gpu" = gpuInfo(platform_idx = as.integer(platform_idx),
                                  device_idx = as.integer(gpu_idx))$double_support,
                  "cpu" = cpuInfo(platform_idx = as.integer(platform_idx),
                                  device_idx = as.integer(gpu_idx))$double_support,
                  stop("Unrecognized device type")
    )
    
    return(out)
    
}

#' @title Set Context
#' @description Change the current context used by default
#' @param id Integer identifying which context to set
#' @seealso \link{listContexts}
#' @export
setContext <- function(id = 1L){
    if(!id %in% listContexts()$context){
        stop("context index not initialized")
    }
    cpp_setContext(id)
}
