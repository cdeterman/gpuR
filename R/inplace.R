
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
                     `+` = gpu_Mat_axpy(1, y, x, inplace = TRUE),
                     `-` = gpu_Mat_axpy(-1, y, x, inplace = TRUE),
                     `*` = gpuMatElemMult(x, y, inplace = TRUE),
                     `/` = gpuMatElemDiv(x, y, inplace = TRUE),
                     stop("undefined operation")
              )
          })

#' @rdname inplace-methods
#' @export
setMethod("inplace",
          signature = c("function", "vclMatrix", "missing"),
          function(f, x, y){
              
              switch(deparse(substitute(f)),
                     `-` = gpuMatrix_unary_axpy(x, inplace = TRUE),
                     `exp` = gpuMatElemExp(x, inplace = TRUE),
                     `abs` = gpuMatElemAbs(x, inplace = TRUE),
                     `sin` = gpuMatElemSin(x, inplace = TRUE),
                     `asin` = gpuMatElemArcSin(x, inplace = TRUE),
                     `sinh` = gpuMatElemHypSin(x, inplace = TRUE),
                     `cos` = gpuMatElemCos(x, inplace = TRUE),
                     `acos` = gpuMatElemArcCos(x, inplace = TRUE),
                     `cosh` = gpuMatElemHypCos(x, inplace = TRUE),
                     `tan` = gpuMatElemTan(x, inplace = TRUE),
                     `atan` = gpuMatElemArcTan(x, inplace = TRUE),
                     `tanh` = gpuMatElemHypTan(x, inplace = TRUE),
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
                     `+` = gpu_Mat_axpy(1, x, y, inplace = TRUE, AisScalar = TRUE),
                     `-` = gpu_Mat_axpy(-1, x, y, inplace = TRUE, AisScalar = TRUE),
                     `/` = gpuMatScalarDiv(x, y, AisScalar = TRUE, inplace = TRUE),
                     `*` = gpuMatScalarMult(y, x, inplace = TRUE),
                     stop("undefined operation")
              )
          })

#' @rdname inplace-methods
#' @export
setMethod("inplace",
          signature = c("function", "vclMatrix", "numeric"),
          function(f, x, y){
              
              switch(deparse(substitute(f)),
                     `+` = gpu_Mat_axpy(1, x, y, inplace = TRUE, BisScalar = TRUE),
                     `-` = gpu_Mat_axpy(-1, x, y, inplace = TRUE, BisScalar = TRUE),
                     `/` = gpuMatScalarDiv(x, y, inplace = TRUE),
                     `*` = gpuMatScalarMult(x, y, inplace = TRUE),
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
                     `*` = gpuMatElemMult(x, y, inplace = TRUE),
                     `/` = gpuMatElemDiv(x, y, inplace = TRUE),
                     stop("undefined operation")
              )
          })


#' @rdname inplace-methods
#' @export
setMethod("inplace",
          signature = c("function", "gpuMatrix", "missing"),
          function(f, x, y){
              
              switch(deparse(substitute(f)),
                     `-` = gpuMatrix_unary_axpy(x, inplace = TRUE),
                     `exp` = gpuMatElemExp(x, inplace = TRUE),
                     `abs` = gpuMatElemAbs(x, inplace = TRUE),
                     `sin` = gpuMatElemSin(x, inplace = TRUE),
                     `asin` = gpuMatElemArcSin(x, inplace = TRUE),
                     `sinh` = gpuMatElemHypSin(x, inplace = TRUE),
                     `cos` = gpuMatElemCos(x, inplace = TRUE),
                     `acos` = gpuMatElemArcCos(x, inplace = TRUE),
                     `cosh` = gpuMatElemHypCos(x, inplace = TRUE),
                     `tan` = gpuMatElemTan(x, inplace = TRUE),
                     `atan` = gpuMatElemArcTan(x, inplace = TRUE),
                     `tanh` = gpuMatElemHypTan(x, inplace = TRUE),
                     stop("undefined operation")
              )
          })

#' @rdname inplace-methods
#' @export
setMethod("inplace",
          signature = c("function", "numeric", "gpuMatrix"),
          function(f, x, y){
              
              switch(deparse(substitute(f)),
                     `+` = gpu_Mat_axpy(1, x, y, inplace = TRUE, AisScalar = TRUE),
                     `-` = gpu_Mat_axpy(-1, x, y, inplace = TRUE, AisScalar = TRUE),
                     `/` = gpuMatScalarDiv(x, y, AisScalar = TRUE, inplace = TRUE),
                     `*` = gpuMatScalarMult(y, x, inplace = TRUE),
                     stop("undefined operation")
              )
          })

#' @rdname inplace-methods
#' @export
setMethod("inplace",
          signature = c("function", "gpuMatrix", "numeric"),
          function(f, x, y){
              
              switch(deparse(substitute(f)),
                     `+` = gpu_Mat_axpy(1, x, y, inplace = TRUE, BisScalar = TRUE),
                     `-` = gpu_Mat_axpy(-1, x, y, inplace = TRUE, BisScalar = TRUE),
                     `/` = gpuMatScalarDiv(x, y, inplace = TRUE),
                     `*` = gpuMatScalarMult(x, y, inplace = TRUE),
                     stop("undefined operation")
              )
          })


#' @rdname inplace-methods
#' @export
setMethod("inplace",
          signature = c("function", "vclVector", "vclVector"),
          function(f, x, y){
              
              switch(deparse(substitute(f)),
                     `+` = gpuVec_axpy(1, y, x, inplace = TRUE),
                     `-` = gpuVec_axpy(-1, y, x, inplace = TRUE),
                     `*` = gpuVecElemMult(x, y, inplace = TRUE),
                     `/` = gpuVecElemDiv(x, y, inplace = TRUE),
                     stop("undefined operation")
              )
          })


#' @rdname inplace-methods
#' @export
setMethod("inplace",
          signature = c("function", "vclVector", "missing"),
          function(f, x, y){
              
              switch(deparse(substitute(f)),
                     `exp` = gpuVecElemExp(x, inplace = TRUE),
                     `abs` = gpuVecElemAbs(x, inplace = TRUE),
                     `sin` = gpuVecElemSin(x, inplace = TRUE),
                     `asin` = gpuVecElemArcSin(x, inplace = TRUE),
                     `sinh` = gpuVecElemHypSin(x, inplace = TRUE),
                     `cos` = gpuVecElemCos(x, inplace = TRUE),
                     `acos` = gpuVecElemArcCos(x, inplace = TRUE),
                     `cosh` = gpuVecElemHypCos(x, inplace = TRUE),
                     `tan` = gpuVecElemTan(x, inplace = TRUE),
                     `atan` = gpuVecElemArcTan(x, inplace = TRUE),
                     `tanh` = gpuVecElemHypTan(x, inplace = TRUE),
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
		  	           gpuVec_axpy(1, z, x, inplace = TRUE)
		  	       },
		  	       `-` = {
		  	           z <- vclVector(rep(y, length(x)), type=typeof(x), ctx_id = x@.context_index)
		  	           gpuVec_axpy(-1, z, x, inplace = TRUE)
		  	       },
		  	       `*` = gpuVecScalarMult(x, y, inplace = TRUE),
		  	       `/` = gpuVecScalarDiv(x, y, inplace = TRUE),
		  		   stop("undefined operation")
		  	)
		  })

#' @rdname inplace-methods
#' @export
setMethod("inplace",
          signature = c("function", "numeric", "vclVector"),
          function(f, x, y){
              
              switch(deparse(substitute(f)),
                     `+` = {
                         z <- vclVector(x, length = length(y), type=typeof(y), ctx_id = y@.context_index)
                         gpuVec_axpy(1, z, y, inplace = TRUE)
                     },
                     `-` = {
                         z <- vclVector(x, length = length(y), type=typeof(y), ctx_id = y@.context_index)
                         gpuVec_axpy(-1, z, y, inplace = TRUE, order = 1)
                     },
                     `*` = gpuVecScalarMult(x, y, inplace = TRUE),
                     `/` = gpuVecScalarDiv(x, y, 1, inplace = TRUE),
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
                     `*` = gpuVecElemMult(x, y, inplace = TRUE),
                     `/` = gpuVecElemDiv(x, y, inplace = TRUE),
                     stop("undefined operation")
              )
          })

#' @rdname inplace-methods
#' @export
setMethod("inplace",
          signature = c("function", "gpuVector", "missing"),
          function(f, x, y){
              
              switch(deparse(substitute(f)),
                     `exp` = gpuVecElemExp(x, inplace = TRUE),
                     `abs` = gpuVecElemAbs(x, inplace = TRUE),
                     `sin` = gpuVecElemSin(x, inplace = TRUE),
                     `asin` = gpuVecElemArcSin(x, inplace = TRUE),
                     `sinh` = gpuVecElemHypSin(x, inplace = TRUE),
                     `cos` = gpuVecElemCos(x, inplace = TRUE),
                     `acos` = gpuVecElemArcCos(x, inplace = TRUE),
                     `cosh` = gpuVecElemHypCos(x, inplace = TRUE),
                     `tan` = gpuVecElemTan(x, inplace = TRUE),
                     `atan` = gpuVecElemArcTan(x, inplace = TRUE),
                     `tanh` = gpuVecElemHypTan(x, inplace = TRUE),
                     stop("undefined operation")
              )
          })


#' @rdname inplace-methods
#' @export
setMethod("inplace",
          signature = c("function", "gpuVector", "numeric"),
          function(f, x, y){
              
              switch(deparse(substitute(f)),
                     `+` = {
                         z <- gpuVector(y, length = length(x), type=typeof(x), ctx_id = x@.context_index)
                         gpuVec_axpy(1, z, x, inplace = TRUE)
                     },
                     `-` = {
                         z <- gpuVector(y, length = length(x), type=typeof(x), ctx_id = x@.context_index)
                         gpuVec_axpy(-1, z, x, inplace = TRUE)
                     },
                     `*` = gpuVecScalarMult(x, y, inplace = TRUE),
                     `/` = gpuVecScalarDiv(x, y, inplace = TRUE),
                     stop("undefined operation")
              )
          })


#' @rdname inplace-methods
#' @export
setMethod("inplace",
          signature = c("function", "numeric", "gpuVector"),
          function(f, x, y){
              
              switch(deparse(substitute(f)),
                     `+` = {
                         z <- gpuVector(x, length = length(y), type=typeof(y), ctx_id = y@.context_index)
                         gpuVec_axpy(1, z, y, inplace = TRUE)
                     },
                     `-` = {
                         z <- gpuVector(x, length = length(y), type=typeof(y), ctx_id = y@.context_index)
                         gpuVec_axpy(-1, z, y, inplace = TRUE, order = 1)
                     },
                     `*` = gpuVecScalarMult(x, y, inplace = TRUE),
                     `/` = gpuVecScalarDiv(x, y, 1, inplace = TRUE),
                     stop("undefined operation")
              )
          })


