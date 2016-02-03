.onLoad <- function(libname, pkgname) {
    options(gpuR.print.warning=TRUE)
    options(gpuR.default.type = "double")
    options(gpuR.default.device.type = "gpu")
    
    # Initialize all possible contexts
    if (!identical(Sys.getenv("APPVEYOR"), "True")) {
        # initialize contexts and return default device
#         default_device <- initContexts()
#         packageStartupMessage(paste0("gpuR 1.1.0\nDefault device: ", default_device))
    }
}

.onUnload <- function(libpath) {
    options(gpuR.print.warning=NULL)
    options(gpuR.default.type = NULL)
    options(gpuR.default.device.type = NULL)
}
