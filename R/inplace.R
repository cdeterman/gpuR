
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
              
              switch(deparse(substitute(f)),
                     `+` = vclMat_axpy(1, y, x, inplace = TRUE),
                     `-` = vclMat_axpy(-1, y, x, inplace = TRUE),
                     `*` = vclMatElemMult(x, y, inplace = TRUE),
                     stop("undefined operation")
              )
          })

#' @rdname inplace-methods
#' @export
setMethod("inplace",
          signature = c("function", "vclMatrix", "missing"),
          function(f, x, y){
              
              switch(deparse(substitute(f)),
                     `-` = vclMatrix_unary_axpy(x, inplace = TRUE),
                     `exp` = vclMatElemExp(x, inplace = TRUE),
                     `abs` = vclMatElemAbs(x, inplace = TRUE),
                     stop("undefined operation")
              )
          })

# @rdname inplace-methods
# @export
# setMethod("inplace",
# 		  signature = c("function", "vclMatrix", "vclVector", "vclVector"),
# 		  function(f, x, y){
# 
# 		  	switch(deparse(substitute(f)),
# 		  		   `-` = vclMatrix_unary_axpy(x, inplace = TRUE),
# 		  		   `exp` = vclMatElemExp(x, inplace = TRUE),
# 		  		   `abs` = vclMatElemAbs(x, inplace = TRUE),
# 		  		   stop("undefined operation")
# 		  	)
# 		  })

#' @rdname inplace-methods
#' @export
setMethod("inplace",
          signature = c("function", "numeric", "vclMatrix"),
          function(f, x, y){

              switch(deparse(substitute(f)),
                     `+` = vclMat_axpy(1, x, y, inplace = TRUE, AisScalar = TRUE),
                     `-` = vclMat_axpy(-1, y, x, inplace = TRUE, BisScalar = TRUE),
                     `/` = {
                         # x = vclMatrix(x, ncol=ncol(y), nrow=nrow(y), type=typeof(y), ctx_id = y@.context_index)
                         vclMatScalarDiv(x, y, AisScalar = TRUE, inplace = TRUE)
                     },
                     `*` = vclMatScalarMult(y, x, inplace = TRUE),
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
          signature = c("function", "vclVector", "missing"),
          function(f, x, y){
              
              switch(deparse(substitute(f)),
                     `abs` = vclVecElemAbs(x, inplace = TRUE),
                     stop("undefined operation")
              )
          })

#' @rdname inplace-methods
#' @export
setMethod("inplace",
		  signature = c("function", "vclVector", "numeric"),
		  function(f, x, y){
		  	
		  	switch(deparse(substitute(f)),
		  		   `+` = {
		  		   		z <- vclVector(rep(y, length(x)), type=typeof(x), ctx_id = x@.context_index)
		  		   		vclVec_axpy(1, z, x, inplace = TRUE)
		  		   	},
		  		   `*` = vclVecScalarMult(x, y, TRUE),
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




