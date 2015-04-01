.onLoad <- function(libname, pkgname) {
    options(bigGPU.default.type = "integer")
}

.onUnload <- function(libpath) {
    options(bigGPU.default.type = NULL)
}