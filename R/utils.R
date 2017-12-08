
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


#' @title Permuting functions for \code{gpuR} objects
#' @description Generate a permutation of row or column indices
#' @param x A \code{gpuR} matrix object
#' @param MARGIN dimension over which the ordering should be applied, 1
#' indicates rows, 2 indicates columns
#' @param order An integer vector indicating order of rows to assign
#' @return A \code{gpuR} object
#' @author Charles Determan Jr.
#' @docType methods
#' @rdname permute-methods
#' @export
permute <- function(x, MARGIN, order) UseMethod("permute") 
 
 
#'@export
permute.vclMatrix <- function(x, MARGIN = 1, order){
    
    assert_is_scalar(MARGIN)
    if(MARGIN != 1){
        stop("only row permuting currently available")
    }
    
    assert_is_not_null(order)
    
    type <- typeof(x)
    
    file <- switch(type,
                   "float" = system.file("CL", "fset_row_order2.cl", package = "gpuR"),
                   "double" = system.file("CL", "dset_row_order2.cl", package = "gpuR"),
                   stop("only float and double type currently supported")
    )
    
    if(!file_test("-f", file)){
        stop("kernel file does not exist")
    }
    kernel <- readChar(file, file.info(file)$size)
    
    maxWorkGroupSize <- 
        switch(deviceType(x@.device_index, x@.context_index),
               "gpu" = gpuInfo(x@.device_index, x@.context_index)$maxWorkGroupSize,
               "cpu" = cpuInfo(x@.device_index, x@.context_index)$maxWorkGroupSize,
               stop("unrecognized device type")
        )
    
    # Y <- vclMatrix(nrow = nrow(x), ncol = ncol(x), type = type, ctx_id = x@.context_index)
    
    switch(type,
           "float" = cpp_vclMatrix_set_row_order(x@address, 
                                                 TRUE,
                                                 order - 1,
                                                 kernel,
                                                 sqrt(maxWorkGroupSize),
                                                 6L,
                                                 x@.context_index - 1),
           "double" = cpp_vclMatrix_set_row_order(x@address, 
                                                  TRUE,
                                                  order - 1,
                                                  kernel,
                                                  sqrt(maxWorkGroupSize),
                                                  8L,
                                                  x@.context_index - 1),
           stop("only float and double currently supported"))
    
    
    return(invisible(x))
    
}


#'@export
permute.vclVector <- function(x, MARGIN = 1, order){
    assert_is_not_null(order)
    
    type <- typeof(x)
    
    file <- switch(type,
                   "float" = system.file("CL", "fPermute.cl", package = "gpuR"),
                   "double" = system.file("CL", "dPermute.cl", package = "gpuR"),
                   stop("only float and double type currently supported")
    )
    
    if(!file_test("-f", file)){
        stop("kernel file does not exist")
    }
    kernel <- readChar(file, file.info(file)$size)
    
    ctx_id <- x@.context_index - 1
    
    switch(type,
           "float" = cpp_vclVector_permute(x@address,
                                           order - 1,
                                           kernel,
                                           6L,
                                           ctx_id),
           "double" = cpp_vclVector_permute(x@address,
                                            order - 1,
                                            kernel,
                                            8L,
                                            ctx_id),
           stop("only float and double currently supported"))
    
    
    return(invisible(x))
}

#' @title Row and Column Names
#' @description Retrieve or set the row or column names of a gpuR matrix object
#' @param x A gpuR matrix object
#' @param do.NULL logical. If \code{FALSE} names are NULL, names are created. 
#' (not currently used)
#' @param prefix for create names. (not currently used)
#' @param value A character vector to assign as row/column names
#' @param ... Additional arguments
#' @docType methods
#' @rdname colnames-methods
#' @export
colnames <- function(x, do.NULL, prefix) UseMethod("colnames")

#' @rdname colnames-methods
#' @export
colnames.default <- base::colnames

#' @rdname colnames-methods
#' @export
colnames.gpuMatrix <- function(x, ...)
{
    type <- switch(typeof(x),
                   "integer" = 4L,
                   "float" = 6L,
                   "double" = 8L
    )

    cnames <- getCols(x@address, type)
    
    if(length(cnames) == 0){
        cnames <- NULL
    }

    return(cnames)
}

#' @rdname colnames-methods
#' @export
setMethod("colnames<-",
          signature = "gpuMatrix",
          function(x, value)
          {

              assert_is_character(value)

              type <- switch(typeof(x),
                             "integer" = 4L,
                             "float" = 6L,
                             "double" = 8L
              )

              setCols(x@address, value, type)

              return(invisible(x))
          })


#' @rdname colnames-methods
#' @export
colnames.vclMatrix <- function(x, ...)
{
    type <- switch(typeof(x),
                   "integer" = 4L,
                   "float" = 6L,
                   "double" = 8L
    )
    
    cnames <- getVCLcols(x@address, type)
    
    if(length(cnames) == 0){
        cnames <- NULL
    }
    
    return(cnames)
}

#' @rdname colnames-methods
#' @export
setMethod("colnames<-",
          signature = "vclMatrix",
          function(x, value)
          {
              
              assert_is_character(value)
              
              type <- switch(typeof(x),
                             "integer" = 4L,
                             "float" = 6L,
                             "double" = 8L
              )
              
              setVCLcols(x@address, value, type)
              
              return(invisible(x))
          })
