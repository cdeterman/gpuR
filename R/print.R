#' @title S3 print for gpuMatrix objects
#' @description prints a gpuMatrix object that is truncated to fit the screen
#' @param x A gpuMatrix object
#' @param ... Additional arguments to print
#' @param n Number of rows to display
#' @param width Number of columns to display
#' @export
print.gpuMatrix <- function(x, ..., n = NULL, width = NULL) {
    cat("Source: gpuR Matrix ", dim_desc(x), "\n", sep = "")
    cat("\n")
    
    if(!is.null(n)){
        assert_is_integer(n)   
    }else{
        n <- ifelse(nrow(x) >= 5, 5L, nrow(x))
    }
    if(!is.null(width)){
        assert_is_integer(width)    
    }else{
        width <- ifelse(ncol(x) >= 5, 5L, ncol(x))
    }
    
    if(width > ncol(x)) stop("width greater than number of columns")
    
    tab <- switch(typeof(x),
                  "integer" = truncIntgpuMat(x@address, n, width),
                  "float" = truncFloatgpuMat(x@address, n, width),
                  "double" = truncDoublegpuMat(x@address, n, width)
    )
    
    block = structure(
        list(table = tab, extra = ncol(x)-width), 
        class = "trunc_gpuTable")
    
    print(block)    
    
    invisible(x)
}


print.trunc_gpuTable <- function(x, ...) {
    if (!is.null(x$table)) {
        print(x$table)
    }
    
    if (x$extra > 0) {
        nvars <- x$extra
        cat("\n", paste0(nvars, " variables not shown"), "\n", sep = "")
    }
    
    invisible()
}
