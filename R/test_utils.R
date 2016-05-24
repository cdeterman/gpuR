###################################
### Unit Test Utility Functions ###
###################################

# The following functions are simply used to facilitate
# the unit tests implemented by this package.  For example, the user
# may install this package with the correct drivers but not have any
# valid GPU devices or a valid GPU may not support double precision.
# These functions will allow some tests to be skipped so that all
# relevant functions can be evaluated.

# check if any GPUs can be found
#' @title Skip test for GPUs
#' @description Function to skip testthat tests
#' if no valid GPU's are detected
#' @export
has_gpu_skip <- function() {
    gpuCheck <- try(detectGPUs(), silent=TRUE)
    if(class(gpuCheck)[1] == "try-error"){
        testthat::skip("No GPUs available")
    }else{
        if (gpuCheck == 0) {
            testthat::skip("No GPUs available")
        }
    }
}

# check if multiple GPUs can be found
#' @title Skip test in less than 2 GPUs
#' @description Function to skip testthat tests
#' if less than 2 valid GPU's are detected
#' @export
has_multiple_gpu_skip <- function() {
    gpuCheck <- try(detectGPUs(), silent=TRUE)
    if(class(gpuCheck)[1] == "try-error"){
        testthat::skip("No GPUs available")
    }else{
        if (gpuCheck < 2) {
            testthat::skip("Only one GPU available")
        }
    }
}

# check if any CPUs can be found
#' @title Skip test for CPUs
#' @description Function to skip testthat tests
#' if no valid CPU's are detected
#' @export
has_cpu_skip <- function() {
    cpuCheck <- try(detectCPUs(), silent=TRUE)
    if(class(cpuCheck)[1] == "try-error"){
        testthat::skip("No CPUs available")
    }else{
        if (cpuCheck == 0) {
            testthat::skip("No CPUs available")
        }
    }
}

# check if GPU supports double precision
#' @title Skip test for GPU double precision
#' @description Function to skip testthat tests
#' if the detected GPU doesn't support double precision
#' @export
has_double_skip <- function() {
    deviceCheck <- try(deviceHasDouble(), silent=TRUE)
    if(class(deviceCheck)[1] == "try-error"){
        testthat::skip("Default device doesn't have double precision")
    }else{
        if (!deviceCheck) {
            testthat::skip("Default device doesn't support double precision")
        }
    }
}

# check if multiple GPUs supports double precision
#' @title Skip test for multiple GPUs with double precision
#' @description Function to skip testthat tests
#' if their aren't multiple detected GPU with double precision
#' @export
has_multiple_double_skip <- function() {
    
    contexts <- listContexts()
    gpus_with_double = 0
    
    for(i in seq(nrow(contexts))){
        gpuCheck <- try(
            deviceHasDouble(contexts$platform_index[i] + 1L, 
                            contexts$device_index[i] + 1L)
            , silent=TRUE)
        if(class(gpuCheck)[1] == "try-error"){
            next
        }else{
            if (!gpuCheck) {
                # This device doesn't support double precision
            }else{
                gpus_with_double = gpus_with_double + 1
            }
        }
    }
    
    if(gpus_with_double < 2){
        testthat::skip("Less than 2 GPUs with double precision")
    }
}
