.onLoad <- function(libname, pkgname) {
    options(gpuR.print.warning=TRUE)
    options(gpuR.default.type = "double")
    options(gpuR.default.device = "gpu")
}

.onUnload <- function(libpath) {
    options(gpuR.print.warning=NULL)
    options(gpuR.default.type = NULL)
    options(gpuR.default.device = NULL)
}
