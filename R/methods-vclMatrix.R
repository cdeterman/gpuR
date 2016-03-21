
#' @rdname extract-methods
#' @export
setMethod("[",
          signature(x = "vclMatrix", i = "missing", j = "missing", drop = "missing"),
          function(x, i, j, drop) {
              switch(typeof(x),
                     "integer" = return(VCLtoMatSEXP(x@address, 4L)),
                     "float" = return(VCLtoMatSEXP(x@address, 6L)),
                     "double" = return(VCLtoMatSEXP(x@address, 8L))
              )
          })

#' @rdname extract-methods
#' @export
setMethod("[",
          signature(x = "vclMatrix", i = "missing", j = "numeric", drop="missing"),
          function(x, i, j, drop) {
              switch(typeof(x),
                     "integer" = return(vclGetCol(x@address, j, 4L)),
                     "float" = return(vclGetCol(x@address, j, 6L)),
                     "double" = return(vclGetCol(x@address, j, 8L))
              )
          })


#' @rdname extract-methods
#' @export
setMethod("[",
          signature(x = "vclMatrix", i = "numeric", j = "missing", drop="missing"),
          function(x, i, j, drop) {
              switch(typeof(x),
                     "integer" = return(vclGetRow(x@address, i, 4L)),
                     "float" = return(vclGetRow(x@address, i, 6L)),
                     "double" = return(vclGetRow(x@address, i, 8L))
              )
          })

#' @rdname extract-methods
#' @export
setMethod("[",
          signature(x = "vclMatrix", i = "numeric", j = "numeric", drop="missing"),
          function(x, i, j, drop) {
              switch(typeof(x),
                     "integer" = return(vclGetElement(x@address, i, j, 4L)),
                     "float" = return(vclGetElement(x@address, i, j, 6L)),
                     "double" = return(vclGetElement(x@address, i, j, 8L))
              )
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
                     "double" = vclSetCol(x@address, j, value, 8L)
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
                     "integer" = vclSetCol(x@address, j, value, 4L)
              )
              return(x)
          })

#' @rdname extract-methods
#' @export
setMethod("[<-",
          signature(x = "vclMatrix", i = "numeric", j = "missing", value = "numeric"),
          function(x, i, j, value) {
              
              if(length(value) != ncol(x)){
                  stop("number of items to replace is not a multiple of replacement length")
              }
              
              if(i > nrow(x)){
                  stop("row index exceeds number of rows")
              }
              
              switch(typeof(x),
                     "float" = vclSetRow(x@address, i, value, 6L),
                     "double" = vclSetRow(x@address, i, value, 8L)
              )
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
                     "integer" = vclSetRow(x@address, i, value, 4L)
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
                            
              switch(typeof(x),
                     "float" = vclSetElement(x@address, i, j, value, 6L),
                     "double" = vclSetElement(x@address, i, j, value, 8L)
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
              
              switch(typeof(x),
                     "integer" = vclSetElement(x@address, i, j, value, 4L)
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
                         e2 <- vclMatrix(e2, ncol=ncol(e1), nrow=nrow(e1), type=typeof(e1))
                         vclMat_axpy(1, e1, e2)
                     },
                     `-` = {
                         e2 <- vclMatrix(e2, ncol=ncol(e1), nrow=nrow(e1), type=typeof(e1))
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
                         e1 = vclMatrix(e1, ncol=ncol(e2), nrow=nrow(e2), type=typeof(e2))
                         vclMat_axpy(1, e1, e2)
                     },
                     `-` = {
                         e1 = vclMatrix(e1, ncol=ncol(e2), nrow=nrow(e2), type=typeof(e2))
                         vclMat_axpy(-1, e2, e1)
                     },
                     `*` = vclMatScalarMult(e2, e1),
                     `/` = {
                         e1 = vclMatrix(e1, ncol=ncol(e2), nrow=nrow(e2), type=typeof(e2))
                         vclMatElemDiv(e1, e2)
                     },
                     `^` = {
                         e1 <- vclMatrix(e1, ncol=ncol(e2), nrow=nrow(e2), type=typeof(e2))
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
              switch(typeof(x),
                     "integer" = return(vcl_inrow(x@address)),
                     "float" = return(vcl_fnrow(x@address)),
                     "double" = return(vcl_dnrow(x@address))
              )
          }
)

#' @rdname nrow-gpuR
#' @export
setMethod('ncol', signature(x="vclMatrix"),
          function(x) {
              switch(typeof(x),
                     "integer" = return(vcl_incol(x@address)),
                     "float" = return(vcl_fncol(x@address)),
                     "double" = return(vcl_dncol(x@address))
              )
          }
)


#' @rdname dim-methods
#' @aliases dim-vclMatrix
#' @export
setMethod('dim', signature(x="vclMatrix"),
          function(x) return(c(nrow(x), ncol(x))))



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
              device_flag <- 
                  switch(options("gpuR.default.device.type")$gpuR.default.device.type,
                         "cpu" = 1, 
                         "gpu" = 0,
                         stop("unrecognized default device option"
                         )
                  )
              
              type = typeof(x)
              
              if( type == "integer"){
                  stop("Integer type not currently supported")
              }
              
              D <- vclMatrix(nrow=nrow(x), ncol=nrow(x), type=type)
              
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
              
              device_flag <- 
                  switch(options("gpuR.default.device.type")$gpuR.default.device.type,
                         "cpu" = 1, 
                         "gpu" = 0,
                         stop("unrecognized default device option"
                         )
                  )
              
              type = typeof(x)
              
              if( type == "integer"){
                  stop("Integer type not currently supported")
              }
              
              D <- vclMatrix(nrow=nrow(x), ncol=nrow(y), type=type)
              
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
                                            address = cpp_deepcopy_vclMatrix(object@address, 4L),
                            								.context_index = object@.context_index,
                            								.platform_index = object@.platform_index,
                            								.platform = object@.platform,
                            								.device_index = object@.device_index,
                            								.device = object@.device),
                            "float" = new("fvclMatrix", 
                                          address = cpp_deepcopy_vclMatrix(object@address, 6L),
                            							.context_index = object@.context_index,
                            							.platform_index = object@.platform_index,
                            							.platform = object@.platform,
                            							.device_index = object@.device_index,
                            							.device = object@.device),
                            "double" = new("dvclMatrix", 
                                           address = cpp_deepcopy_vclMatrix(object@address, 8L),
                            							 .context_index = object@.context_index,
                            							 .platform_index = object@.platform_index,
                            							 .platform = object@.platform,
                            							 .device_index = object@.device_index,
                            							 .device = object@.device),
                            stop("unrecognized type")
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
              
              device_flag <- 
                  switch(options("gpuR.default.device.type")$gpuR.default.device.type,
                         "cpu" = 1, 
                         "gpu" = 0,
                         stop("unrecognized default device option"
                         )
                  )
              
              ptr <- switch(typeof(x),
                            "integer" = {
                                address <- cpp_cbind_vclMatrix(x@address, y@address, 4L, device_flag)
                                new("ivclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "float" = {
                                address <- cpp_cbind_vclMatrix(x@address, y@address, 6L, device_flag)
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
                                address <- cpp_cbind_vclMatrix(x@address, y@address, 8L, device_flag)
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
              
              device_flag <- 
                  switch(options("gpuR.default.device.type")$gpuR.default.device.type,
                         "cpu" = 1, 
                         "gpu" = 0,
                         stop("unrecognized default device option"
                         )
                  )
              
              x <- vclMatrix(x, nrow=nrow(y), ncol=1, type=typeof(y))
              
              ptr <- switch(typeof(x),
                            "integer" = {
                                address <- cpp_cbind_vclMatrix(x@address, y@address, 4L, device_flag)
                                new("ivclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "float" = {
                                address <- cpp_cbind_vclMatrix(x@address, y@address, 6L, device_flag)
                                new("fvclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "double" = {
                                address <- cpp_cbind_vclMatrix(x@address, y@address, 8L, device_flag)
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
              
              device_flag <- 
                  switch(options("gpuR.default.device.type")$gpuR.default.device.type,
                         "cpu" = 1, 
                         "gpu" = 0,
                         stop("unrecognized default device option"
                         )
                  )
              
              y <- vclMatrix(y, nrow=nrow(x), ncol=1, type=typeof(x))
              
              ptr <- switch(typeof(x),
                            "integer" = {
                                address <- cpp_cbind_vclMatrix(x@address, y@address, 4L, device_flag)
                                new("ivclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "float" = {
                                address <- cpp_cbind_vclMatrix(x@address, y@address, 6L, device_flag)
                                new("fvclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "double" = {
                                address <- cpp_cbind_vclMatrix(x@address, y@address, 8L, device_flag)
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
              
              device_flag <- 
                  switch(options("gpuR.default.device.type")$gpuR.default.device.type,
                         "cpu" = 1, 
                         "gpu" = 0,
                         stop("unrecognized default device option"
                         )
                  )
              
              ptr <- switch(typeof(x),
                            "integer" = {
                                address <- cpp_rbind_vclMatrix(x@address, y@address, 4L, device_flag)
                                new("ivclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "float" = {
                                address <- cpp_rbind_vclMatrix(x@address, y@address, 6L, device_flag)
                                new("fvclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "double" = {
                                address <- cpp_rbind_vclMatrix(x@address, y@address, 8L, device_flag)
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
              
              device_flag <- 
                  switch(options("gpuR.default.device.type")$gpuR.default.device.type,
                         "cpu" = 1, 
                         "gpu" = 0,
                         stop("unrecognized default device option"
                         )
                  )
              
              x <- vclMatrix(x, nrow=1, ncol=ncol(y), type=typeof(y))
              
              ptr <- switch(typeof(x),
                            "integer" = {
                                address <- cpp_rbind_vclMatrix(x@address, y@address, 4L, device_flag)
                                new("ivclMatrix",
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "float" = {
                                address <- cpp_rbind_vclMatrix(x@address, y@address, 6L, device_flag)
                                new("fvclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "double" = {
                                address <- cpp_rbind_vclMatrix(x@address, y@address, 8L, device_flag)
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
              
              device_flag <- 
                  switch(options("gpuR.default.device.type")$gpuR.default.device.type,
                         "cpu" = 1, 
                         "gpu" = 0,
                         stop("unrecognized default device option"
                         )
                  )
              
              y <- vclMatrix(y, nrow=1, ncol=ncol(x), type=typeof(x))
              
              ptr <- switch(typeof(x),
                            "integer" = {
                                address <- cpp_rbind_vclMatrix(x@address, y@address, 4L, device_flag)
                                new("ivclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "float" = {
                                address <- cpp_rbind_vclMatrix(x@address, y@address, 6L, device_flag)
                                new("fvclMatrix", 
                                    address = address,
                                    .context_index = x@.context_index,
                                    .platform_index = x@.platform_index,
                                    .platform = x@.platform,
                                    .device_index = x@.device_index,
                                    .device = x@.device)
                            },
                            "double" = {
                                address <- cpp_rbind_vclMatrix(x@address, y@address, 8L, device_flag)
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

setMethod("t", c(x = "vclMatrix"),
          function(x){
              return(vclMatrix_t(x))
          }
)
