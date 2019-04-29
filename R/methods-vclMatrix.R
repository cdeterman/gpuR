

#' @export
as.matrix.vclMatrix <- function(x, ...){
    out <- x[]
    return(out)
} 


#' @rdname grapes-times-grapes-methods
#' @export
setMethod("%*%", signature(x="vclMatrix", y = "vclMatrix"),
          function(x,y)
          {
              if( dim(x)[2] != dim(y)[1]){
                  stop("Non-conformant matrices")
              }
              return(vclMatMult(x, y))
          },
          valueClass = "vclMatrix"
)

#' @rdname grapes-times-grapes-methods
#' @export
setMethod("%*%", signature(x="vclMatrix", y = "vclVector"),
          function(x,y)
          {
              if( ncol(x) != length(y)){
                  stop("Non-conformable arguments")
              }
              return(vclGEMV(x, y))
          },
          valueClass = "vclVector"
)

#' @rdname grapes-times-grapes-methods
#' @export
setMethod("%*%", signature(x="vclMatrix", y = "matrix"),
          function(x,y)
          {
              if( dim(x)[2] != dim(y)[1]){
                  stop("Non-conformant matrices")
              }
              y <- vclMatrix(y, type = typeof(x), ctx_id = x@.context_index)
              return(vclMatMult(x, y))
          },
          valueClass = "vclMatrix"
)

#' @rdname grapes-times-grapes-methods
#' @export
setMethod("%*%", signature(x="matrix", y = "vclMatrix"),
          function(x,y)
          {
              if( dim(x)[2] != dim(y)[1]){
                  stop("Non-conformant matrices")
              }
              x <- vclMatrix(x, type = typeof(y), ctx_id = y@.context_index)
              return(vclMatMult(x, y))
          },
          valueClass = "vclMatrix"
)

#' @rdname Arith-methods
#' @export
setMethod("Arith", c(e1="vclMatrix", e2="vclMatrix"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              
              switch(op,
                     `+` = gpu_Mat_axpy(1, e1, e2),
                     `-` = gpu_Mat_axpy(-1, e2, e1),
                     `*` = gpuMatElemMult(e1, e2),
                     `/` = gpuMatElemDiv(e1,e2),
                     `^` = gpuMatElemPow(e1, e2),
                     stop("undefined operation")
              )
          },
          valueClass = "vclMatrix"
)

#' @rdname Arith-methods
#' @export
setMethod("Arith", c(e1="vclMatrix", e2="matrix"),
          function(e1, e2)
          {
              
              e2 <- vclMatrix(e2, type = typeof(e1), ctx_id = e1@.context_index)
              
              op = .Generic[[1]]
              
              switch(op,
                     `+` = gpu_Mat_axpy(1, e1, e2),
                     `-` = gpu_Mat_axpy(-1, e2, e1),
                     `*` = gpuMatElemMult(e1, e2),
                     `/` = gpuMatElemDiv(e1,e2),
                     `^` = gpuMatElemPow(e1, e2),
                     stop("undefined operation")
              )
          },
          valueClass = "vclMatrix"
)

#' @rdname Arith-methods
#' @export
setMethod("Arith", c(e1="matrix", e2="vclMatrix"),
          function(e1, e2)
          {
              e1 <- vclMatrix(e1, type = typeof(e2), ctx_id = e2@.context_index)
              
              op = .Generic[[1]]
              
              switch(op,
                     `+` = gpu_Mat_axpy(1, e1, e2),
                     `-` = gpu_Mat_axpy(-1, e2, e1),
                     `*` = gpuMatElemMult(e1, e2),
                     `/` = gpuMatElemDiv(e1,e2),
                     `^` = gpuMatElemPow(e1, e2),
                     stop("undefined operation")
              )
          },
          valueClass = "vclMatrix"
)

#' @rdname Arith-methods
#' @export
setMethod("Arith", c(e1="vclMatrix", e2="numeric"),
          function(e1, e2)
          {
              assert_is_of_length(e2, 1)

              op = .Generic[[1]]
              switch(op,
                     `+` = {
                         e2 <- vclMatrix(e2, ncol=ncol(e1), nrow=nrow(e1), type=typeof(e1), ctx_id=e1@.context_index)
                         gpu_Mat_axpy(1, e1, e2)
                     },
                     `-` = {
                         e2 <- vclMatrix(e2, ncol=ncol(e1), nrow=nrow(e1), type=typeof(e1), ctx_id=e1@.context_index)
                         gpu_Mat_axpy(-1, e2, e1)
                     },
                     `*` = gpuMatScalarMult(e1, e2),
                     `/` = gpuMatScalarDiv(e1, e2),
                     `^` = gpuMatScalarPow(e1, e2),
                     stop("undefined operation")
              )

          },
          valueClass = "vclMatrix"
)

#' @rdname Arith-methods
#' @export
setMethod("Arith", c(e1="numeric", e2="vclMatrix"),
          function(e1, e2)
          {
              assert_is_of_length(e1, 1)

              op = .Generic[[1]]
              switch(op,
                     `+` = {
                         # e1 = vclMatrix(e1, ncol=ncol(e2), nrow=nrow(e2), type=typeof(e2), ctx_id = e2@.context_index)
                         gpu_Mat_axpy(1, e1, e2, AisScalar = TRUE)
                     },
                     `-` = {
                         # e1 = vclMatrix(e1, ncol=ncol(e2), nrow=nrow(e2), type=typeof(e2), ctx_id = e2@.context_index)
                         # vclMat_axpy(-1, e2, e1)
                         gpu_Mat_axpy(-1, e1, e2, AisScalar = TRUE)
                     },
                     `*` = gpuMatScalarMult(e2, e1),
                     `/` = {
                         # e1 = vclMatrix(e1, ncol=ncol(e2), nrow=nrow(e2), type=typeof(e2), ctx_id = e2@.context_index)
                         gpuMatScalarDiv(e1, e2, AisScalar = TRUE)
                     },
                     `^` = {
                         e1 <- vclMatrix(e1, ncol=ncol(e2), nrow=nrow(e2), type=typeof(e2), ctx_id = e2@.context_index)
                         gpuMatElemPow(e1, e2)
                     },
                     stop("undefined operation")
              )
          },
          valueClass = "vclMatrix"
)

#' @rdname Arith-methods
#' @export
setMethod("Arith", c(e1="vclMatrix", e2="missing"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              switch(op,
                     `-` = gpuMatrix_unary_axpy(e1),
                     stop("undefined operation")
              )
          },
          valueClass = "vclMatrix"
)

#' @rdname Arith-methods
#' @export
setMethod("Arith", c(e1="vclMatrix", e2="vclVector"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              
              switch(op,
                     `+` = gpuMatVec_axpy(1, e1, e2),
                     `-` = gpuMatVec_axpy(-1, e2, e1),
                     stop("undefined operation")
              )
          },
          valueClass = "vclMatrix"
)


#' @rdname Math-methods
#' @export
setMethod("Math", c(x="vclMatrix"),
          function(x)
          {
              op = .Generic[[1]]
              switch(op,
                     `sin` = gpuMatElemSin(x),
                     `asin` = gpuMatElemArcSin(x),
                     `sinh` = gpuMatElemHypSin(x),
                     `cos` = gpuMatElemCos(x),
                     `acos` = gpuMatElemArcCos(x),
                     `cosh` = gpuMatElemHypCos(x),
                     `tan` = gpuMatElemTan(x),
                     `atan` = gpuMatElemArcTan(x),
                     `tanh` = gpuMatElemHypTan(x),
                     `log10` = gpuMatElemLog10(x),
                     `exp` = gpuMatElemExp(x),
                     `abs` = gpuMatElemAbs(x),
                     `sqrt` = gpuMatSqrt(x),
                     `sign` = gpuMatSign(x),
                     stop("undefined operation")
              )
          },
          valueClass = "vclMatrix"
)

#' @rdname log-methods
#' @export
setMethod("log", c(x="vclMatrix"),
          function(x, base=NULL)
          {
              if(is.null(base)){
                  gpuMatElemLog(x) 
              }else{
                  assert_is_numeric(base)
                  gpuMatElemLogBase(x, base)
              }
              
          },
          valueClass = "vclMatrix"
)



#' @rdname nrow-gpuR
#' @export
setMethod('nrow', signature(x="vclMatrix"), 
          function(x) {
              
              result <- switch(typeof(x),
                     "integer" = cpp_vcl_nrow(x@address, 4L),
                     "float" = cpp_vcl_nrow(x@address, 6L),
                     "double" = cpp_vcl_nrow(x@address, 8L),
                     "fcomplex" = cpp_vcl_nrow(x@address, 10L),
                     "dcomplex" = cpp_vcl_nrow(x@address, 12L),
                     stop("unsupported matrix type")
              )
              
              return(result)
          }
)

#' @rdname nrow-gpuR
#' @export
setMethod('ncol', signature(x="vclMatrix"),
          function(x) {
              
              result <- switch(typeof(x),
                     "integer" = cpp_vcl_ncol(x@address, 4L),
                     "float" = cpp_vcl_ncol(x@address, 6L),
                     "double" = cpp_vcl_ncol(x@address, 8L),
                     "fcomplex" = cpp_vcl_ncol(x@address, 10L),
                     "dcomplex" = cpp_vcl_ncol(x@address, 12L),
                     stop("unsupported matrix type")
              )
              
              return(result)
          }
)


#' @rdname dim-methods
#' @aliases dim-vclMatrix
#' @export
setMethod("dim", signature(x="vclMatrix"),
          function(x) {
              return(c(nrow(x), ncol(x)))
          }
)


#' @rdname length-methods
#' @aliases length-vclMatrix
#' @export
setMethod("length", signature(x="vclMatrix"),
          function(x) {
              return(nrow(x) *ncol(x))
          }
)



#' @title vclMatrix Crossproduct
#' @description Return the matrix cross-product of two conformable
#' matrices using a GPU.  This is equivalent to t(x) %*% y (crossprod)
#' or x %*% t(y) (tcrossprod) but faster as no data transfer between
#' device and host is required.
#' @param x A vclMatrix
#' @param y A vclMatrix
#' @return A vclMatrix
#' @author Charles Determan Jr.
#' @docType methods
#' @rdname vclMatrix-crossprod
#' @aliases crossprod,vclMatrix
#' @export
setMethod("crossprod",
          signature(x = "vclMatrix", y = "missing"),
          function(x, y){
              vcl_crossprod(x, x)
          })


#' @rdname vclMatrix-crossprod
#' @export
setMethod("crossprod",
          signature(x = "vclMatrix", y = "vclMatrix"),
          function(x, y){
              vcl_crossprod(x, y)
          })

#' @rdname vclMatrix-crossprod
#' @export
setMethod("crossprod",
          signature(x = "vclMatrix", y = "matrix"),
          function(x, y){
              y <- vclMatrix(y, type = typeof(x), ctx_id = x@.context_index)
              vcl_crossprod(x, y)
          })

#' @rdname vclMatrix-crossprod
#' @export
setMethod("crossprod",
          signature(x = "matrix", y = "vclMatrix"),
          function(x, y){
              x <- vclMatrix(x, type = typeof(y), ctx_id = y@.context_index)
              vcl_crossprod(x, y)
          })

#' @rdname vclMatrix-crossprod
#' @export
setMethod("crossprod",
          signature(x = "vclMatrix", y = "vclVector"),
          function(x, y){
              vcl_mat_vec_crossprod(x, y)
          })


#' @rdname vclMatrix-crossprod
#' @export
setMethod("crossprod",
          signature(x = "vclVector", y = "vclMatrix"),
          function(x, y){
              vcl_mat_vec_crossprod(x, y)
          })


#' @rdname vclMatrix-crossprod
setMethod("tcrossprod",
          signature(x = "vclMatrix", y = "missing"),
          function(x, y){
              vcl_tcrossprod(x, x)
          })


#' @rdname vclMatrix-crossprod
#' @export
setMethod("tcrossprod",
          signature(x = "vclMatrix", y = "vclMatrix"),
          function(x, y){
              vcl_tcrossprod(x, y)
          })

#' @rdname vclMatrix-crossprod
#' @export
setMethod("tcrossprod",
          signature(x = "matrix", y = "vclMatrix"),
          function(x, y){
              x <- vclMatrix(x, type = typeof(y), ctx_id = y@.context_index)
              vcl_tcrossprod(x, y)
          })

#' @rdname vclMatrix-crossprod
#' @export
setMethod("tcrossprod",
          signature(x = "vclMatrix", y = "matrix"),
          function(x, y){
              y <- vclMatrix(y, type = typeof(x), ctx_id = x@.context_index)
              vcl_tcrossprod(x, y)
          })

#' @rdname vclMatrix-crossprod
#' @export
setMethod("tcrossprod",
          signature(x = "vclMatrix", y = "vclVector"),
          function(x, y){
              vcl_mat_vec_tcrossprod(x, y)
          })

#' @rdname vclMatrix-crossprod
#' @export
setMethod("tcrossprod",
          signature(x = "vclVector", y = "vclMatrix"),
          function(x, y){
              vcl_mat_vec_tcrossprod(x, y)
          })

#' @rdname cov-methods
#' @export
setMethod("cov",
          signature(x = "vclMatrix", y = "missing", use = "missing", method = "missing"),
          function(x, y = NULL, use = NULL, method = "pearson") {
              if(method != "pearson"){
                  stop("Only pearson covariance implemented")
              }
              return(vclMatrix_pmcc(x))
          })

#' @rdname cov-methods
#' @export
setMethod("cov",
          signature(x = "vclMatrix", y = "vclMatrix", use = "missing", method = "missing"),
          function(x, y = NULL, use = NULL, method = "pearson") {
              if(method != "pearson"){
                  stop("Only pearson covariance implemented")
              }
              return(vclMatrix_pmcc(x, y))
          })

#' @rdname cov-methods
#' @export
setMethod("cov",
          signature(x = "vclMatrix", y = "missing", use = "missing", method = "character"),
          function(x, y = NULL, use = NULL, method = "pearson") {
              if(method != "pearson"){
                  stop("Only pearson covariance implemented")
              }
              return(vclMatrix_pmcc(x))
          })

#' @rdname cov-methods
#' @export
setMethod("cov",
          signature(x = "vclMatrix", y = "vclMatrix", use = "missing", method = "character"),
          function(x, y = NULL, use = NULL, method = "pearson") {
              if(method != "pearson"){
                  stop("Only pearson covariance implemented")
              }
              return(vclMatrix_pmcc(x, y))
          })

#' @title Row and Column Sums and Means of vclMatrix
#' @description Row and column sums and of vclMatrix objects
#' @param x A vclMatrix object
#' @return A gpuVector object
#' @author Charles Determan Jr.
#' @docType methods
#' @rdname vclMatrix.colSums
#' @aliases colSums,vclMatrix
#' @aliases rowSums,vclMatrix
#' @export
setMethod("colSums",
          signature(x = "vclMatrix"),
          function(x){
              vclMatrix_colSums(x)
          })


#' @rdname vclMatrix.colSums
#' @export
setMethod("rowSums",
          signature(x = "vclMatrix"),
          function(x){
              vclMatrix_rowSums(x)
          })



#' @rdname vclMatrix.colSums
#' @export
setMethod("colMeans",
          signature(x = "vclMatrix"),
          function(x){
              vclMatrix_colMeans(x)
          })


#' @rdname vclMatrix.colSums
#' @export
setMethod("rowMeans",
          signature(x = "vclMatrix"),
          function(x){
              vclMatrix_rowMeans(x)
          })


#' @rdname Summary-methods
#' @export
setMethod("Summary", c(x="vclMatrix"),
          function(x, ..., na.rm)
          {              
              op = .Generic
              result <- switch(op,
                               `max` = vclMatMax(x),
                               `min` = vclMatMin(x),
                               `sum` = vclMatSum(x),
                               stop("undefined operation")
              )
              return(result)
          }
)

#' @title GPU Distance Matrix Computations
#' @description This function computes and returns the distance matrix 
#' computed by using the specified distance measure to compute the distances 
#' between the rows of a data matrix.
#' @param x A gpuMatrix or vclMatrix object
#' @param y A gpuMatrix or vclMatrix object
#' @param method the distance measure to be used. This must be one of
#' "euclidean" or "sqEuclidean".
#' @param diag logical value indicating whether the diagonal of the distance 
#' matrix should be printed
#' @param upper logical value indicating whether the upper triangle of the 
#' distance matrix
#' @param p The power of the Minkowski distance (not currently used)
#' @return a gpuMatrix/vclMatrix containing the corresponding distances
#' @rdname dist-vclMatrix
#' @aliases dist,vclMatrix
#' @export
setMethod("dist", signature(x="vclMatrix"),
          function(x, method = "euclidean", diag = FALSE, upper = FALSE, p = 2)
          {
              
              type = typeof(x)
              
              # if( type == "integer"){
              #     stop("Integer type not currently supported")
              # }
              
              D <- vclMatrix(nrow=nrow(x), ncol=nrow(x), type=type, ctx_id=x@.context_index)

              switch(method,
                     "euclidean" = vclMatrix_euclidean(
                         x, 
                         D,
                         diag,
                         upper,
                         p,
                         FALSE),
                     "sqEuclidean" = vclMatrix_euclidean(
                         x, 
                         D,
                         diag,
                         upper,
                         p,
                         TRUE),
                     stop("method not currently supported")
              )
              
             return(D)
          }
)

#' @rdname dist-vclMatrix
#' @aliases distance,vclMatrix
setMethod("distance", signature(x = "vclMatrix", y = "vclMatrix"),
          function(x, y, method = "euclidean")
          {
              if(identical(x, y)){
                  same <- TRUE
                  warning("x is the same as y, did you mean to use 'dist' instead?")
              }else{
                  same <- FALSE
              }
              
              if(ncol(x) != ncol(y)){
                  stop("columns in x and y are not equivalent")
              }
              
              type = typeof(x)
              
              # if( type == "integer"){
              #     stop("Integer type not currently supported")
              # }
              
              assert_are_identical(x@.context_index, y@.context_index)
              
              D <- vclMatrix(nrow=nrow(x), ncol=nrow(y), type=type, ctx_id = x@.context_index)

              switch(method,
                     "euclidean" = vclMatrix_peuclidean(
                         x, 
                         y,
                         D,
                         FALSE),
                     "sqEuclidean" = vclMatrix_peuclidean(
                         x, 
                         y,
                         D,
                         TRUE),
                     stop("method not currently supported")
              )
              
              if(same){
                  for(i in 1:ncol(D)){
                      D[i,i] <- 0
                  }
              }
              
              return(D)
          }
)

#' @rdname gpuR-deepcopy
setMethod("deepcopy", signature(object ="vclMatrix"),
          function(object, source = FALSE){
              
              out <- switch(typeof(object),
                            "integer" = new("ivclMatrix",
                                            address = cpp_deepcopy_vclMatrix(object@address, 4L, object@.context_index - 1,
                                                                             source),
										.context_index = object@.context_index,
										.platform_index = object@.platform_index,
										.platform = object@.platform,
										.device_index = object@.device_index,
										.device = object@.device),
                            "float" = new("fvclMatrix", 
                                          address = cpp_deepcopy_vclMatrix(object@address, 6L, object@.context_index - 1,
                                                                           source),
                            							.context_index = object@.context_index,
                            							.platform_index = object@.platform_index,
                            							.platform = object@.platform,
                            							.device_index = object@.device_index,
                            							.device = object@.device),
                            "double" = new("dvclMatrix", 
                                           address = cpp_deepcopy_vclMatrix(object@address, 8L, object@.context_index - 1,
                                                                            source),
                            							 .context_index = object@.context_index,
                            							 .platform_index = object@.platform_index,
                            							 .platform = object@.platform,
                            							 .device_index = object@.device_index,
                            							 .device = object@.device),
                            stop("unsupported matrix type")
              )
              
              return(out)
          })

#' @rdname gpuR-block
setMethod("block",
          signature(object = "vclMatrix", 
                    rowStart = "integer", rowEnd = "integer",
                    colStart = "integer", colEnd = "integer"),
          function(object, rowStart, rowEnd, colStart, colEnd){
              
              assert_all_are_positive(c(rowStart, rowEnd, colStart, colEnd))
              assert_all_are_in_range(c(rowStart, rowEnd), lower = 1, upper = nrow(object)+1)
              assert_all_are_in_range(c(colStart, colEnd), lower = 1, upper = ncol(object)+1)
              
              ptr <- switch(typeof(object),
                            "float" = {
                                address <- cpp_vclMatrix_block(object@address, rowStart-1, rowEnd, colStart-1, colEnd, 6L)
                                new("fvclMatrixBlock", 
                                    address = address,
                                    .context_index = object@.context_index,
                                    .platform_index = object@.platform_index,
                                    .platform = object@.platform,
                                    .device_index = object@.device_index,
                                    .device = object@.device)
                            },
                            "double" = {
                                address <- cpp_vclMatrix_block(object@address, rowStart-1, rowEnd, colStart-1, colEnd, 8L)
                                new("dvclMatrixBlock", 
                                    address = address,
                                    .context_index = object@.context_index,
                                    .platform_index = object@.platform_index,
                                    .platform = object@.platform,
                                    .device_index = object@.device_index,
                                    .device = object@.device)
                            },
                            stop("type not recognized")
              )
              
              return(ptr)
              
          })

setMethod("cbind2",
          signature(x = "vclMatrix", y = "vclMatrix"),
          function(x, y, ...){
              if(nrow(x) != nrow(y)){
                  stop("number of rows of matrices must match")
              }
              
              assert_are_identical(x@.context_index, y@.context_index)
              
              z <- vclMatrix(nrow = nrow(x), ncol = ncol(x) + ncol(y), type = typeof(y), y@.context_index)
              
              cbind_wrapper(x,y,z)
              
              return(z)
          })

setMethod("cbind2",
          signature(x = "numeric", y = "vclMatrix"),
          function(x, y, ...){
              
              x <- vclMatrix(x, nrow=nrow(y), ncol=1, type=typeof(y), y@.context_index)
              z <- vclMatrix(nrow = nrow(y), ncol = 1 + ncol(y), type = typeof(y), y@.context_index)
              
              cbind_wrapper(x,y,z)
              
              return(z)
          })

setMethod("cbind2",
          signature(x = "vclMatrix", y = "numeric"),
          function(x, y, ...){
              
              y <- vclMatrix(y, nrow=nrow(x), ncol=1, type=typeof(x), x@.context_index)
              z <- vclMatrix(nrow = nrow(x), ncol = 1 + ncol(x), type = typeof(x), ctx_id = x@.context_index)
              
              cbind_wrapper(x,y,z)
              
              return(z)
          })


setMethod("cbind2",
          signature(x = "vclMatrix", y = "vclVector"),
          function(x, y, ...){
              if(nrow(x) != length(y)){
                  stop("number of rows of matrices must match")
              }
              
              # print('cbind called')
              assert_are_identical(x@.context_index, y@.context_index)
              
              z <- vclMatrix(nrow = nrow(x), ncol = ncol(x) + 1, type = typeof(y), ctx_id = y@.context_index)
              
              z[,1:(ncol(z) - 1)] <- x
              z[,ncol(z)] <- y
              
              # cbind_wrapper2(x,y,z, FALSE)
              
              # print(z[])
              # 
              # print('cbind complete')
              
              return(z)
          })


setMethod("cbind2",
          signature(x = "vclVector", y = "vclMatrix"),
          function(x, y, ...){
              if(length(x) != nrow(y)){
                  stop("number of rows of matrices must match")
              }
              
              assert_are_identical(x@.context_index, y@.context_index)
              
              z <- vclMatrix(nrow = nrow(y), ncol = 1 + ncol(y), type = typeof(y), ctx_id = y@.context_index)
              
              cbind_wrapper2(y,x,z, TRUE)
              
              return(z)
          })


setMethod("rbind2",
          signature(x = "vclMatrix", y = "vclMatrix"),
          function(x, y, ...){
              if(ncol(x) != ncol(y)){
                  stop("number of columns of matrices must match")
              }
              
              assert_are_identical(x@.context_index, y@.context_index)
              
              ptr <- switch(typeof(x),
                            "integer" = {
                                address <- cpp_rbind_vclMatrix(x@address, y@address, 4L, x@.context_index - 1)
                                new("ivclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "float" = {
                                address <- cpp_rbind_vclMatrix(x@address, y@address, 6L, x@.context_index - 1)
                                new("fvclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "double" = {
                                address <- cpp_rbind_vclMatrix(x@address, y@address, 8L, x@.context_index - 1)
                                new("dvclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            stop("type not recognized")
              )
              
              return(ptr)
          })

setMethod("rbind2",
          signature(x = "numeric", y = "vclMatrix"),
          function(x, y, ...){
              
              x <- vclMatrix(x, nrow=1, ncol=ncol(y), type=typeof(y), y@.context_index)
              
              ptr <- switch(typeof(x),
                            "integer" = {
                                address <- cpp_rbind_vclMatrix(x@address, y@address, 4L, y@.context_index - 1)
                                new("ivclMatrix",
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "float" = {
                                address <- cpp_rbind_vclMatrix(x@address, y@address, 6L, y@.context_index - 1)
                                new("fvclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "double" = {
                                address <- cpp_rbind_vclMatrix(x@address, y@address, 8L, y@.context_index - 1)
                                new("dvclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)                            },
                            stop("type not recognized")
              )
              
              return(ptr)
          })

setMethod("rbind2",
          signature(x = "vclMatrix", y = "numeric"),
          function(x, y, ...){
              
              y <- vclMatrix(y, nrow=1, ncol=ncol(x), type=typeof(x), x@.context_index)
              
              ptr <- switch(typeof(x),
                            "integer" = {
                                address <- cpp_rbind_vclMatrix(x@address, y@address, 4L, x@.context_index - 1)
                                new("ivclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "float" = {
                                address <- cpp_rbind_vclMatrix(x@address, y@address, 6L, x@.context_index - 1)
                                new("fvclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "double" = {
                                address <- cpp_rbind_vclMatrix(x@address, y@address, 8L, x@.context_index - 1)
                                new("dvclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            stop("type not recognized")
              )
              
              return(ptr)
          })


#' @rdname t-methods
#' @aliases t,vclMatrix
#' @export
setMethod("t", c(x = "vclMatrix"),
          function(x){
              return(vclMatrix_t(x))
          }
)


#' @title gpuR Matrix Diagonals
#' @description Extract or replace the diagonal of a matrix
#' @param x A gpuR matrix object
#' @param value A vector object (gpuR)
#' @return A gpuR vector object of the matrix diagonal of \code{x}.  The 
#' replacement form returns nothing as it replaces the diagonal of \code{x}.
#' @note If an identity matrix is desired, please see \link{identity_matrix}.
#' @author Charles Determan Jr.
#' @seealso \link{identity_matrix}
#' @rdname diag-methods
#' @aliases diag,vclMatrix
#' @export
setMethod("diag", c(x = "vclMatrix"),
          function(x){
              # get diagonal elements
              return(vclMatrix_get_diag(x))
          }
)

#' @rdname diag-methods
#' @aliases diag<-,vclMatrix,vclVector
#' @export
setMethod("diag<-", c(x = "vclMatrix", value = "vclVector"),
          function(x, value){
              
              if(nrow(x) != length(value)){
                  stop("replacement diagnonal has wrong length")
              }
              
              # get diagonal elements
              vclMat_vclVec_set_diag(x, value)
              
              return(invisible(x))
          }
)


#' @title Identity Matrix on Device
#' @description Creates an identity matrix directly on the current device
#' (e.g. GPU)
#' @param x A numeric value indicating the order of the identity matrix
#' @param type A character string specifying the type of gpuMatrix.  Default
#' is derived from \code{getOption("gpuR.default.type")}.
#' @note This function was only created for \code{vclMatrix} objects as
#' the copy from CPU to \code{gpuMatrix} is trivial using the base
#' \link[base]{diag} function.
#' @return A \code{vclMatrix} object
#' @author Charles Determan Jr.
#' @export
identity_matrix <- function(x, type = NULL){
    
    assert_is_a_number(x)
    
    if(is.null(type)){
        type <- getOption("gpuR.default.type")
    }
    
    iMat <- vclMatrix(nrow = x, ncol = x, type = type)
    
    switch(type,
           "integer" = cpp_identity_vclMatrix(iMat@address, 4L),
           "float" = cpp_identity_vclMatrix(iMat@address, 6L),
           "double" = cpp_identity_vclMatrix(iMat@address, 8L)
    )
    
    return(iMat)
}


#' @title Calculate Determinant of a Matrix on GPU
#' @description \code{det} calculates the determinant of a matrix.
#' @param x A gpuR matrix object
#' @return The determine of \code{x}
#' @note This function uses an LU decomposition and the \code{det} 
#' function is simply a wrapper returning the determinant product
#' @author Charles Determan Jr.
#' @rdname det-methods
#' @aliases det,vclMatrix
#' @export
setMethod("det", c(x = "vclMatrix"),
          function(x){
              return(gpuMat_det(x))
          }
)

