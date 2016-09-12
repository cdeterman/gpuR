#' @importFrom utils strOptions packageVersion tail

.onLoad <- function(libname, pkgname) {
    options(gpuR.print.warning=TRUE)
    options(gpuR.default.type = "double")
    # options(gpuR.default.device.type = "gpu")
}

.onAttach <- function(libname, pkgname) {
    # Initialize all possible contexts
    if (!identical(Sys.getenv("APPVEYOR"), "True") && !identical(Sys.getenv("TRAVIS"), "true")) {
        # initialize contexts
        # default_device <- initContexts()
        initContexts()
        # print("context initialization successful")
        # packageStartupMessage(paste0("gpuR ", packageVersion('gpuR'), "\nDefault device: ", default_device))
        packageStartupMessage(paste0("gpuR ", packageVersion('gpuR')))
        # print("startup message not problem")
    }
}

.onUnload <- function(libpath) {
    options(gpuR.print.warning=NULL)
    options(gpuR.default.type = NULL)
    # options(gpuR.default.device.type = NULL)
}
