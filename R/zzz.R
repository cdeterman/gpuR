.onLoad <- function(libname, pkgname) {
    options(gpuR.default.type = "integer")
}

.onUnload <- function(libpath) {
    options(gpuR.default.type = NULL)
}