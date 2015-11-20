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
    gpuCheck <- try(deviceHasDouble(), silent=TRUE)
    if(class(gpuCheck)[1] == "try-error"){
        testthat::skip("No GPUs available")
    }else{
        if (!gpuCheck) {
            testthat::skip("GPU doesn't support double precision")
        }
    }
}
