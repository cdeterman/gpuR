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
#' @export
has_gpu_skip <- function() {
    gpuCheck <- try(detectGPUs(), silent=TRUE)
    if(class(gpuCheck)[1] == "try-error"){
        skip("No GPUs available")
    }else{
        if (gpuCheck == 0) {
            skip("No GPUs available")
        }
    }
}

# check if GPU supports double precision
#' @export
has_double_skip <- function() {
    gpuCheck <- try(deviceHasDouble(), silent=TRUE)
    if(class(gpuCheck)[1] == "try-error"){
        skip("No GPUs available")
    }else{
        if (!gpuCheck) {
            skip("GPU doesn't support double precision")
        }
    }
}