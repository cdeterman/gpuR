

#' @export
as.matrix.vclMatrix <- function(x, ...){
    out <- x[]
    return(out)
} 


#' @rdname extract-methods
#' @export
setMethod("[",
          signature(x = "vclMatrix", i = "missing", j = "missing", drop = "missing"),
          function(x, i, j, drop) {
              
              Rmat <- switch(typeof(x),
                     "integer" = VCLtoMatSEXP(x@address, 4L),
                     "float" = VCLtoMatSEXP(x@address, 6L),
                     "double" = VCLtoMatSEXP(x@address, 8L),
                     stop("unsupported matrix type")
              )
              
	      return(Rmat)

          })

#' @rdname extract-methods
#' @export
setMethod("[",
          signature(x = "vclMatrix", i = "missing", j = "numeric", drop="missing"),
          function(x, i, j, drop) {
              
              Rmat <- switch(typeof(x),
                     "integer" = vclGetCol(x@address, j, 4L, x@.context_index - 1),
                     "float" = vclGetCol(x@address, j, 6L, x@.context_index - 1),
                     "double" = vclGetCol(x@address, j, 8L, x@.context_index - 1),
                     stop("unsupported matrix type")
              )
              
	      return(Rmat)

          })


#' @rdname extract-methods
#' @export
setMethod("[",
          signature(x = "vclMatrix", i = "numeric", j = "missing", drop="missing"),
          function(x, i, j, ..., drop) {
              
              if(tail(i, 1) > length(x)){
                  stop("Index out of bounds")
              }
              
              type <- switch(typeof(x),
                             "integer" = 4L,
                             "float" = 6L,
                             "double" = 8L,
                             stop("type not recognized")
              )
              
              if(nargs() == 3){
                  return(vclGetRow(x@address, i, type, x@.context_index - 1))
              }else{
                  
                  output <- vector(ifelse(type == 4L, "integer", "numeric"), length(i))
                  
                  nr <- nrow(x)
                  col_idx <- 1
                  for(elem in seq_along(i)){
                      if(i[elem] > nr){
                          tmp <- ceiling(i[elem]/nr)
                          if(tmp != col_idx){
                              col_idx <- tmp
                          }
                          
                          row_idx <- i[elem] - (nr * (col_idx - 1))
                          
                      }else{
                          row_idx <- i[elem]
                      }
                      
                      output[elem] <- vclGetElement(x@address, row_idx, col_idx, type)
                  }
                  
                  return(output)
              }
              
              # Rmat <- switch(typeof(x),
              #        "integer" = vclGetRow(x@address, i, 4L, x@.context_index - 1),
              #        "float" = vclGetRow(x@address, i, 6L, x@.context_index - 1),
              #        "double" = vclGetRow(x@address, i, 8L, x@.context_index - 1),
              #        stop("unsupported matrix type")
              # )
              
	      # return(Rmat)

          })

#' @rdname extract-methods
#' @export
setMethod("[",
          signature(x = "vclMatrix", i = "numeric", j = "numeric", drop="missing"),
          function(x, i, j, drop) {
              
              Rmat <- switch(typeof(x),
                     "integer" = vclGetElement(x@address, i, j, 4L),
                     "float" = vclGetElement(x@address, i, j, 6L),
                     "double" = vclGetElement(x@address, i, j, 8L),
                     stop("unsupported matrix type")
              )
              
	      return(Rmat)

          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "vclMatrix", i = "missing", j = "numeric", value = "numeric"),
          function(x, i, j, value) {
              
              if(length(value) != nrow(x)){
                  stop("number of items to replace is not a multiple of replacement length")
              }
              
              if(j > ncol(x)){
                  stop("column index exceeds number of columns")
              }

              switch(typeof(x),
                     "float" = vclSetCol(x@address, j, value, 6L),
                     "double" = vclSetCol(x@address, j, value, 8L),
                     stop("unsupported matrix type")
              )
              
              return(x)
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "ivclMatrix", i = "missing", j = "numeric", value = "integer"),
          function(x, i, j, value) {
              
              if(length(value) != nrow(x)){
                  stop("number of items to replace is not a multiple of replacement length")
              }
              
              if(j > ncol(x)){
                  stop("column index exceeds number of columns")
              }

              switch(typeof(x),
                     "integer" = vclSetCol(x@address, j, value, 4L),
                     stop("unsupported matrix type")
              )
              
              return(x)
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "vclMatrix", i = "numeric", j = "missing", value = "numeric"),
          function(x, i, j, ..., value) {
              
              assert_all_are_in_closed_range(i, lower = 1, upper = nrow(x))
              
              type <- switch(typeof(x),
                             "integer" = 4L,
                             "float" = 6L,
                             "double" = 8L,
                             stop("type not recognized")
              )
              
              # print(nargs())
              
              if(nargs() == 4){
                  if(length(value) != ncol(x)){
                      stop("number of items to replace is not a multiple of replacement length")
                  }
                  
                  vclSetRow(x@address, i, value, type)
                  
              }else{
                  if(length(value) != length(i)){
                      if(length(value) == 1){
                          value <- rep(value, length(i))
                      }else{
                          stop("number of items to replace is not a multiple of replacement length")
                      }
                  }
                  
                  nr <- nrow(x)
                  col_idx <- 1
                  for(elem in seq_along(i)){
                      if(i[elem] > nr){
                          tmp <- ceiling(i[elem]/nr)
                          if(tmp != col_idx){
                              col_idx <- tmp
                          }
                          
                          row_idx <- i[elem] - (nr * (col_idx - 1))
                          
                      }else{
                          row_idx <- i[elem]
                      }
                      
                      # print(row_idx)
                      # print(col_idx)
                      
                      vclSetElement(x@address, row_idx, col_idx, value[elem], type)
                  }
              }
              
# 	      if(length(value) != ncol(x)){
# 	          stop("number of items to replace is not a multiple of replacement length")
# 	      }
#               
#           if(i > nrow(x)){
#               stop("row index exceeds number of rows")
#           }
#           
#           switch(typeof(x),
#                  "float" = vclSetRow(x@address, i, value, 6L),
#                  "double" = vclSetRow(x@address, i, value, 8L),
#                  stop("unsupported matrix type")
#           )
          
          return(x)
      })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "ivclMatrix", i = "numeric", j = "missing", value = "integer"),
          function(x, i, j, value) {
              
              if(length(value) != ncol(x)){
                  stop("number of items to replace is not a multiple of replacement length")
              }
              
              if(i > nrow(x)){
                  stop("row index exceeds number of rows")
              }

              switch(typeof(x),
                     "integer" = vclSetRow(x@address, i, value, 4L),
                     stop("unsupported matrix type")
              )
              
              return(x)
          })


#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "vclMatrix", i = "numeric", j = "numeric", value = "numeric"),
          function(x, i, j, value) {
              
              assert_all_are_in_closed_range(i, lower = 1, upper=nrow(x))
              assert_all_are_in_closed_range(j, lower = 1, upper=ncol(x))
	      assert_is_scalar(value)

              switch(typeof(x),
                     "float" = vclSetElement(x@address, i, j, value, 6L),
                     "double" = vclSetElement(x@address, i, j, value, 8L),
                     stop("unsupported matrix type")
              )
              
              return(x)
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "ivclMatrix", i = "numeric", j = "numeric", value = "integer"),
          function(x, i, j, value) {
              
              assert_all_are_in_closed_range(i, lower = 1, upper=nrow(x))
              assert_all_are_in_closed_range(j, lower = 1, upper=ncol(x))
	      assert_is_scalar(value)

              switch(typeof(x),
                     "integer" = vclSetElement(x@address, i, j, value, 4L),
                     stop("unsupported matrix type")
              )
              
              return(x)
          })
 
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

#' @rdname Arith-methods
#' @export
setMethod("Arith", c(e1="vclMatrix", e2="vclMatrix"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              switch(op,
                     `+` = vclMat_axpy(1, e1, e2),
                     `-` = vclMat_axpy(-1, e2, e1),
                     `*` = vclMatElemMult(e1, e2),
                     `/` = vclMatElemDiv(e1,e2),
                     `^` = vclMatElemPow(e1, e2),
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
                         vclMat_axpy(1, e1, e2)
                     },
                     `-` = {
                         e2 <- vclMatrix(e2, ncol=ncol(e1), nrow=nrow(e1), type=typeof(e1), ctx_id=e1@.context_index)
                         vclMat_axpy(-1, e2, e1)
                     },
                     `*` = vclMatScalarMult(e1, e2),
                     `/` = vclMatScalarDiv(e1, e2),
                     `^` = vclMatScalarPow(e1, e2),
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
                         e1 = vclMatrix(e1, ncol=ncol(e2), nrow=nrow(e2), type=typeof(e2), ctx_id = e2@.context_index)
                         vclMat_axpy(1, e1, e2)
                     },
                     `-` = {
                         e1 = vclMatrix(e1, ncol=ncol(e2), nrow=nrow(e2), type=typeof(e2), ctx_id = e2@.context_index)
                         vclMat_axpy(-1, e2, e1)
                     },
                     `*` = vclMatScalarMult(e2, e1),
                     `/` = {
                         e1 = vclMatrix(e1, ncol=ncol(e2), nrow=nrow(e2), type=typeof(e2), ctx_id = e2@.context_index)
                         vclMatElemDiv(e1, e2)
                     },
                     `^` = {
                         e1 <- vclMatrix(e1, ncol=ncol(e2), nrow=nrow(e2), type=typeof(e2), ctx_id = e2@.context_index)
                         vclMatElemPow(e1, e2)
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
                     `-` = vclMatrix_unary_axpy(e1),
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
                     `sin` = vclMatElemSin(x),
                     `asin` = vclMatElemArcSin(x),
                     `sinh` = vclMatElemHypSin(x),
                     `cos` = vclMatElemCos(x),
                     `acos` = vclMatElemArcCos(x),
                     `cosh` = vclMatElemHypCos(x),
                     `tan` = vclMatElemTan(x),
                     `atan` = vclMatElemArcTan(x),
                     `tanh` = vclMatElemHypTan(x),
                     `log10` = vclMatElemLog10(x),
                     `exp` = vclMatElemExp(x),
                     `abs` = vclMatElemAbs(x),
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
                  vclMatElemLog(x) 
              }else{
                  assert_is_numeric(base)
                  vclMatElemLogBase(x, base)
              }
              
          },
          valueClass = "vclMatrix"
)



#' @rdname nrow-gpuR
#' @export
setMethod('nrow', signature(x="vclMatrix"), 
          function(x) {
              
              result <- switch(typeof(x),
                     "integer" = vcl_inrow(x@address),
                     "float" = vcl_fnrow(x@address),
                     "double" = vcl_dnrow(x@address),
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
                     "integer" = vcl_incol(x@address),
                     "float" = vcl_fncol(x@address),
                     "double" = vcl_dncol(x@address),
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
#' or x %*% t(t) (tcrossprod) but faster as no data transfer between
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
          signature(x = "vclMatrix", y = "missing", use = "missing", method = "character"),
          function(x, y = NULL, use = NULL, method = "pearson") {
              if(method != "pearson"){
                  stop("Only pearson covariance implemented")
              }
              return(vclMatrix_pmcc(x))
          })

#' @title Row and Column Sums and Means of vclMatrix
#' @description Row and column sums and of vclMatrix objects
#' @param x A vclMatrix object
#' @param na.rm Not currently used
#' @param dims Not currently used
#' @return A gpuVector object
#' @author Charles Determan Jr.
#' @docType methods
#' @rdname vclMatrix.colSums
#' @aliases colSums,vclMatrix
#' @aliases rowSums,vclMatrix
#' @export
setMethod("colSums",
          signature(x = "vclMatrix", na.rm = "missing", dims = "missing"),
          function(x, na.rm, dims){
              vclMatrix_colSums(x)
          })


#' @rdname vclMatrix.colSums
#' @export
setMethod("rowSums",
          signature(x = "vclMatrix", na.rm = "missing", dims = "missing"),
          function(x, na.rm, dims){
              vclMatrix_rowSums(x)
          })



#' @rdname vclMatrix.colSums
#' @export
setMethod("colMeans",
          signature(x = "vclMatrix", na.rm = "missing", dims = "missing"),
          function(x, na.rm, dims){
              vclMatrix_colMeans(x)
          })


#' @rdname vclMatrix.colSums
#' @export
setMethod("rowMeans",
          signature(x = "vclMatrix", na.rm = "missing", dims = "missing"),
          function(x, na.rm, dims){
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
          function(object){
              
              out <- switch(typeof(object),
                            "integer" = new("ivclMatrix",
                                            address = cpp_deepcopy_vclMatrix(object@address, 4L, object@.context_index - 1),
										.context_index = object@.context_index,
										.platform_index = object@.platform_index,
										.platform = object@.platform,
										.device_index = object@.device_index,
										.device = object@.device),
                            "float" = new("fvclMatrix", 
                                          address = cpp_deepcopy_vclMatrix(object@address, 6L, object@.context_index - 1),
                            							.context_index = object@.context_index,
                            							.platform_index = object@.platform_index,
                            							.platform = object@.platform,
                            							.device_index = object@.device_index,
                            							.device = object@.device),
                            "double" = new("dvclMatrix", 
                                           address = cpp_deepcopy_vclMatrix(object@address, 8L, object@.context_index - 1),
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
              
              ptr <- switch(typeof(x),
                            "integer" = {
                                address <- cpp_cbind_vclMatrix(x@address, y@address, 4L,
                                                               x@.context_index - 1)
                                new("ivclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "float" = {
                                address <- cpp_cbind_vclMatrix(x@address, y@address, 6L,
                                                               x@.context_index - 1)
                                new("fvclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device
                                    )
                            },
                            "double" = {
                                address <- cpp_cbind_vclMatrix(x@address, y@address, 8L,
                                                               x@.context_index - 1)
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

setMethod("cbind2",
          signature(x = "numeric", y = "vclMatrix"),
          function(x, y, ...){
              
              x <- vclMatrix(x, nrow=nrow(y), ncol=1, type=typeof(y), y@.context_index)
              
              ptr <- switch(typeof(x),
                            "integer" = {
                                address <- cpp_cbind_vclMatrix(x@address, y@address, 4L, y@.context_index - 1)
                                new("ivclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "float" = {
                                address <- cpp_cbind_vclMatrix(x@address, y@address, 6L, y@.context_index - 1)
                                new("fvclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "double" = {
                                address <- cpp_cbind_vclMatrix(x@address, y@address, 8L, y@.context_index - 1)
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

setMethod("cbind2",
          signature(x = "vclMatrix", y = "numeric"),
          function(x, y, ...){
              
              y <- vclMatrix(y, nrow=nrow(x), ncol=1, type=typeof(x), x@.context_index)
              
              ptr <- switch(typeof(x),
                            "integer" = {
                                address <- cpp_cbind_vclMatrix(x@address, y@address, 4L, x@.context_index - 1)
                                new("ivclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "float" = {
                                address <- cpp_cbind_vclMatrix(x@address, y@address, 6L, x@.context_index - 1)
                                new("fvclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "double" = {
                                address <- cpp_cbind_vclMatrix(x@address, y@address, 8L, x@.context_index - 1)
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



