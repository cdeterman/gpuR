.onLoad <- function(libname, pkgname) {
    options(gpuR.print.warning=TRUE)
    options(gpuR.default.type = "double")
    options(gpuR.default.device.type = "gpu")
    
    # Initialize all possible contexts
    if (!identical(Sys.getenv("APPVEYOR"), "True")) initContexts()
#     options(gpuR.default.device = 1L)
#     options(gpuR.default.platform = 1L)
}

.onUnload <- function(libpath) {
    options(gpuR.print.warning=NULL)
    options(gpuR.default.type = NULL)
    options(gpuR.default.device.type = NULL)
#     options(gpuR.default.device = NULL)
#     options(gpuR.default.platform = NULL)
}
