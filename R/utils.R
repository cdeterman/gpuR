
# This is simply a reproduction of dim_desc from dplyr
# I like the formatted output but it would be silly to 
# require the user to import the package only for this function
dim_desc <- function(x) {
    d <- dim(x)
    d2 <- format(d, big.mark = ",", justify = "none", trim = TRUE)
    d2[is.na(d)] <- "??"
    
    paste0("[", paste0(d2, collapse = " x "), "]")
}



#'@export
str.gpuMatrix <- function(object, vec.len = strOptions()$vec.len, 
                          digits.d = strOptions()$digits.d, ...)
{
    d <- dim(object)
    type <- typeof(object)
    
    prefix <- switch(type,
                     "double" = "num",
                     "float" = "flt",
                     "integer" = "int")
    
    end <- round(vec.len * 1.25)
    
    elems <- round(object[1:end], digits.d)
    
    rows <- paste(1, d[1], sep=":")
    cols <- paste(1, d[2], sep=":")
    
    ss <- paste0(" ", prefix, paste0(" [", rows, ", ", cols, "] "), paste0(elems, collapse = " "), " ", "...", sep = "")
    cat(ss)
    invisible()
}


#'@export
str.vclMatrix <- function(object, vec.len = strOptions()$vec.len, 
                          digits.d = strOptions()$digits.d, ...)
{
    d <- dim(object)
    type <- typeof(object)
    
    prefix <- switch(type,
                     "double" = "num",
                     "float" = "flt",
                     "integer" = "int")
    
    end <- round(vec.len * 1.25)
    
    elems <- round(object[1:end], digits.d)
    
    rows <- paste(1, d[1], sep=":")
    cols <- paste(1, d[2], sep=":")
    
    ss <- paste0(" ", prefix, paste0(" [", rows, ", ", cols, "] "), paste0(elems, collapse = " "), " ", "...", sep = "")
    cat(ss)
    invisible()
}


#'@export
permute <- function(X, MARGIN = 1, order){
    file <- system.file("CL", "fset_row_order.cl", package = "gpuR")
    
    if(!file_test("-f", file)){
        stop("kernel file does not exist")
    }
    kernel <- readChar(file, file.info(file)$size)
    
    maxWorkGroupSize <- 
        switch(deviceType(X@.platform_index, X@.device_index),
               "gpu" = gpuInfo(X@.platform_index, X@.device_index)$maxWorkGroupSize,
               "cpu" = cpuInfo(X@.platform_index, X@.device_index)$maxWorkGroupSize,
               stop("unrecognized device type")
        )
    
    type <- typeof(X)
    
    Y <- vclMatrix(nrow = nrow(X), ncol = ncol(X), type = type, ctx_id = X@.context_index)
    
    # print(Y[])
    
    cpp_vclMatrix_set_row_order(X@address, 
                                Y@address, 
                                TRUE,
                                TRUE,
                                order - 1,
                                kernel,
                                sqrt(maxWorkGroupSize),
                                6L,
                                X@.context_index - 1)
    
    # print(Y[])
    
    return(Y)
    
}