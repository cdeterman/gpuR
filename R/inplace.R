
#' @title Inplace Function Wrapper
#' @description Applies the provided function in-place on the 
#' first object passed
#' @param f A function
#' @param x A gpuR object
#' @param y A gpuR object
#' @return No return, result applied in-place
#' @docType methods
#' @rdname inplace-methods
#' @author Charles Determan Jr.
#' @export
setGeneric("inplace", function(f, x, y){
    standardGeneric("inplace")
})

#' @rdname inplace-methods
#' @export
setMethod("inplace",
          signature = c("function", "vclMatrix", "vclMatrix"),
          function(f, x, y){
              
              # f <- match.fun(f)
              
              switch(deparse(substitute(f)),
                     `+` = vclMat_axpy(1, y, x, inplace = TRUE),
                     `-` = vclMat_axpy(-1, y, x, inplace = TRUE),
                     stop("undefined operation")
              )
          })

#' @rdname inplace-methods
#' @export
setMethod("inplace",
          signature = c("function", "gpuMatrix", "gpuMatrix"),
          function(f, x, y){
              
              switch(deparse(substitute(f)),
                     `+` = gpu_Mat_axpy(1, y, x, inplace = TRUE),
                     `-` = gpu_Mat_axpy(-1, y, x, inplace = TRUE),
                     stop("undefined operation")
              )
          })



#' @rdname inplace-methods
#' @export
setMethod("inplace",
          signature = c("function", "vclVector", "vclVector"),
          function(f, x, y){
              
              switch(deparse(substitute(f)),
                     `+` = vclVec_axpy(1, y, x, inplace = TRUE),
                     `-` = vclVec_axpy(-1, y, x, inplace = TRUE),
                     stop("undefined operation")
              )
          })


#' @rdname inplace-methods
#' @export
setMethod("inplace",
          signature = c("function", "gpuVector", "gpuVector"),
          function(f, x, y){
              
              switch(deparse(substitute(f)),
                     `+` = gpuVec_axpy(1, y, x, inplace = TRUE),
                     `-` = gpuVec_axpy(-1, y, x, inplace = TRUE),
                     stop("undefined operation")
              )
          })




