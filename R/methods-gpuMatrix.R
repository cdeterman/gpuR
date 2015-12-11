#' @import methods
#' @importFrom utils file_test

#' @title gpuMatrix Multiplication
#' @param x A gpuMatrix object
#' @param y A gpuMatrix object
#' @export
setMethod("%*%", signature(x="gpuMatrix", y = "gpuMatrix"),
          function(x,y)
          {
              if( dim(x)[2] != dim(y)[1]){
                  stop("Non-conformant matrices")
              }
              return(gpu_Mat_mult(x, y))
          },
          valueClass = "gpuMatrix"
)

#' @title gpuMatrix Arith methods
#' @param e1 A gpuMatrix object
#' @param e2 A gpuMatrix object
#' @return A gpuMatrix object
#' @export
setMethod("Arith", c(e1="gpuMatrix", e2="gpuMatrix"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              switch(op,
                     `+` = gpu_Mat_axpy(1, e1, e2),
                     `-` = gpu_Mat_axpy(-1, e2, e1),
                     `*` = gpuMatElemMult(e1, e2),
                     `/` = gpuMatElemDiv(e1, e2),
                     `^` = gpuMatElemPow(e1, e2),
                     stop("undefined operation")
              )
          },
valueClass = "gpuMatrix"
)

#' @title gpuMatrix Arith methods
#' @param e1 A gpuMatrix object
#' @param e2 A numeric object
#' @return A gpuMatrix object
#' @export
setMethod("Arith", c(e1="gpuMatrix", e2="numeric"),
          function(e1, e2)
          {
              assert_is_of_length(e2, 1)
              
              op = .Generic[[1]]
              switch(op,
                     `+` = {
                         e2 <- gpuMatrix(matrix(e2, ncol=ncol(e1), nrow=nrow(e1)), type=typeof(e1))
                         gpu_Mat_axpy(1, e1, e2)
                         },
                     `-` = {
                         e2 <- gpuMatrix(matrix(e2, ncol=ncol(e1), nrow=nrow(e1)), type=typeof(e1))
                         gpu_Mat_axpy(-1, e2, e1)
                         },
                     `*` = gpuMatScalarMult(e1, e2),
                     `/` = gpuMatScalarDiv(e1, e2),
                     `^` = {
                         gpuMatScalarPow(e1, e2)
                         #e2 <- gpuMatrix(matrix(e2, ncol=ncol(e1), nrow=nrow(e1)), type=typeof(e1))
                         #gpuMatElemPow(e1, e2)
                         },
                     stop("undefined operation")
              )
          },
valueClass = "gpuMatrix"
)

#' @title gpuMatrix Arith methods
#' @param e1 A numeric object
#' @param e2 A gpuMatrix object
#' @return A gpuMatrix object
#' @export
setMethod("Arith", c(e1="numeric", e2="gpuMatrix"),
          function(e1, e2)
          {
              assert_is_of_length(e1, 1)
              
              op = .Generic[[1]]
              switch(op,
                     `+` = {
                         e1 = gpuMatrix(matrix(e1, ncol=ncol(e2), nrow=nrow(e2)), type=typeof(e2))
                         gpu_Mat_axpy(1, e1, e2)
                         },
                     `-` = {
                         e1 = gpuMatrix(matrix(e1, ncol=ncol(e2), nrow=nrow(e2)), type=typeof(e2))
                         gpu_Mat_axpy(-1, e2, e1)
                         },
                     `*` = gpuMatScalarMult(e2, e1),
                     `/` = {
                         e1 = gpuMatrix(matrix(e1, ncol=ncol(e2), nrow=nrow(e2)), type=typeof(e2))
                         gpuMatElemDiv(e1, e2)
                         },
                     `^` = {
                         e1 <- gpuMatrix(matrix(e1, ncol=ncol(e2), nrow=nrow(e2)), type=typeof(e2))
                         gpuMatElemPow(e1, e2)
                     },
                     stop("undefined operation")
              )
          },
valueClass = "gpuMatrix"
)

#' @title gpuMatrix Arith methods
#' @param e1 A gpuMatrix object
#' @param e2 missing
#' @return A gpuMatrix object
#' @export
setMethod("Arith", c(e1="gpuMatrix", e2="missing"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              switch(op,
                     `-` = gpuMatrix_unary_axpy(e1),
                     stop("undefined operation")
              )
          },
          valueClass = "gpuMatrix"
)

#' @title gpuMatrix Math methods
#' @param x A gpuMatrix object
#' @return A gpuMatrix object
#' @export
setMethod("Math", c(x="gpuMatrix"),
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
                     stop("undefined operation")
              )
          },
valueClass = "gpuMatrix"
)

#' @title gpuMatrix Logarithms
#' @param x A gpuMatrix object
#' @param base A positive number (complex not currently supported by OpenCL):
#' the base with respect to which logarithms are computed.  Defaults to the
#' natural log.
#' @return A gpuMatrix object
#' @export
setMethod("log", c(x="gpuMatrix"),
          function(x, base=NULL)
          {
              if(is.null(base)){
                  gpuMatElemLog(x) 
              }else{
                  assert_is_numeric(base)
                  gpuMatElemLogBase(x, base)
              }
              
          },
          valueClass = "gpuMatrix"
)


#' @title The Number of Rows/Columns of a gpuMatrix
#' @param x A gpuMatrix object
#' @return An integer of length 1
#' @rdname nrow.gpuMatrix
#' @aliases nrow,gpuMatrix
#' @aliases ncol,gpuMatrix
#' @author Charles Determan Jr.
#' @export
setMethod('nrow', signature(x="gpuMatrix"), 
          function(x) {
              switch(typeof(x),
                     "integer" = return(cpp_inrow(x@address)),
                     "float" = return(cpp_fnrow(x@address)),
                     "double" = return(cpp_dnrow(x@address))
              )
          }
)

#' @rdname nrow.gpuMatrix
#' @export
setMethod('ncol', signature(x="gpuMatrix"),
          function(x) {
              switch(typeof(x),
                     "integer" = return(cpp_incol(x@address)),
                     "float" = return(cpp_fncol(x@address)),
                     "double" = return(cpp_dncol(x@address))
              )
          }
)


#' @title gpuMatrix dim method
#' @param x A gpuMatrix object
#' @return A length 2 vector of the number of rows and columns respectively.
#' @author Charles Determan Jr.
#' @aliases dim,gpuMatrix
#' @export
setMethod('dim', signature(x="gpuMatrix"),
          function(x) return(c(nrow(x), ncol(x))))

#' @title Extract gpuMatrix elements
#' @param x A gpuMatrix object
#' @param i indices specifying rows
#' @param j indices specifying columns
#' @param drop missing
#' @param value data of similar type to be added to gpuMatrix object
#' @aliases [,gpuMatrix
#' @aliases [<-,gpuMatrix
#' @author Charles Determan Jr.
#' @rdname extract-gpuMatrix
#' @export
setMethod("[",
          signature(x = "gpuMatrix", i = "missing", j = "missing", drop = "missing"),
          function(x, i, j, drop) {
              switch(typeof(x),
                     "integer" = return(MatXptrToMatSEXP(x@address, 4L)),
                     "float" = return(MatXptrToMatSEXP(x@address, 6L)),
                     "double" = return(MatXptrToMatSEXP(x@address, 8L))
              )
          })

#' @rdname extract-gpuMatrix
#' @export
setMethod("[",
          signature(x = "gpuMatrix", i = "missing", j = "numeric", drop="missing"),
          function(x, i, j, drop) {
              switch(typeof(x),
                     "integer" = return(GetMatCol(x@address, j, 4L)),
                     "float" = return(GetMatCol(x@address, j, 6L)),
                     "double" = return(GetMatCol(x@address, j, 8L))
              )
          })


#' @rdname extract-gpuMatrix
#' @export
setMethod("[",
          signature(x = "gpuMatrix", i = "numeric", j = "missing", drop="missing"),
          function(x, i, j, drop) {
              switch(typeof(x),
                     "integer" = return(GetMatRow(x@address, i, 4L)),
                     "float" = return(GetMatRow(x@address, i, 6L)),
                     "double" = return(GetMatRow(x@address, i, 8L)),
                     stop("type not recognized")
              )
          })

#' @rdname extract-gpuMatrix
#' @export
setMethod("[",
          signature(x = "gpuMatrix", i = "numeric", j = "numeric", drop="missing"),
          function(x, i, j, drop) {
              switch(typeof(x),
                     "integer" = return(GetMatElement(x@address, i, j, 4L)),
                     "float" = return(GetMatElement(x@address, i, j, 6L)),
                     "double" = return(GetMatElement(x@address, i, j, 8L))
              )
          })

#' @rdname extract-gpuMatrix
#' @export
setMethod("[<-",
          signature(x = "gpuMatrix", i = "numeric", j = "missing", value="numeric"),
          function(x, i, j, value) {
              if(length(value) != ncol(x)){
                  stop("number of items to replace is not a multiple of replacement length")
              }
              
              assert_all_are_in_closed_range(i, lower = 1, upper = nrow(x))
              
              switch(typeof(x),
                     "float" = SetMatRow(x@address, i, value, 6L),
                     "double" = SetMatRow(x@address, i, value, 8L),
                     stop("type not recognized")
              )
              return(x)
          })

#' @rdname extract-gpuMatrix
#' @export
setMethod("[<-",
          signature(x = "igpuMatrix", i = "numeric", j = "missing", value="integer"),
          function(x, i, j, value) {
              if(length(value) != ncol(x)){
                  stop("number of items to replace is not a multiple of replacement length")
              }
              
              assert_all_are_in_closed_range(i, lower = 1, upper = nrow(x))
              
              switch(typeof(x),
                     "integer" = SetMatRow(x@address, i, value, 4L),
                     stop("type not recognized")
              )
              return(x)
          })

#' @rdname extract-gpuMatrix
#' @export
setMethod("[<-",
          signature(x = "gpuMatrix", i = "missing", j = "numeric", value="numeric"),
          function(x, i, j, value) {
              
              if(length(value) != nrow(x)){
                  stop("number of items to replace is not a multiple of replacement length")
              }
              
              assert_all_are_in_closed_range(j, lower = 1, upper = ncol(x))
              
              switch(typeof(x),
                     "float" = SetMatCol(x@address, j, value, 6L),
                     "double" = SetMatCol(x@address, j, value, 8L),
                     stop("type not recognized")
              )
              return(x)
          })

#' @rdname extract-gpuMatrix
#' @export
setMethod("[<-",
          signature(x = "igpuMatrix", i = "missing", j = "numeric", value="integer"),
          function(x, i, j, value) {
              
              if(length(value) != nrow(x)){
                  stop("number of items to replace is not a multiple of replacement length")
              }
              
              assert_all_are_in_closed_range(j, lower = 1, upper = ncol(x))
              
              switch(typeof(x),
                     "integer" = SetMatCol(x@address, j, value, 4L),
                     stop("type not recognized")
              )
              return(x)
          })


#' @rdname extract-gpuMatrix
#' @export
setMethod("[<-",
          signature(x = "gpuMatrix", i = "numeric", j = "numeric", value="numeric"),
          function(x, i, j, value) {
              
              assert_all_are_in_closed_range(i, lower = 1, upper = nrow(x))
              assert_all_are_in_closed_range(j, lower = 1, upper = ncol(x))
              
              switch(typeof(x),
                     "float" = SetMatElement(x@address, i, j, value, 6L),
                     "double" = SetMatElement(x@address, i, j, value, 8L),
                     stop("type not recognized")
              )
              return(x)
          })

#' @rdname extract-gpuMatrix
#' @export
setMethod("[<-",
          signature(x = "igpuMatrix", i = "numeric", j = "numeric", value="integer"),
          function(x, i, j, value) {
              
              assert_all_are_in_closed_range(i, lower = 1, upper = nrow(x))
              assert_all_are_in_closed_range(j, lower = 1, upper = ncol(x))
              
              switch(typeof(x),
                     "integer" = SetMatElement(x@address, i, j, value, 4L),
                     stop("type not recognized")
              )
              return(x)
          })

#' @title Row and Column Sums and Means of gpuMatrix
#' @description Row and column sums and of gpuMatrix objects
#' @param x A gpuMatrix object
#' @param na.rm Not currently used
#' @param dims Not currently used
#' @return A gpuVector object
#' @author Charles Determan Jr.
#' @docType methods
#' @rdname gpuMatrix.colSums
#' @aliases colSums,gpuMatrix
#' @aliases rowSums,gpuMatrix
#' @export
setMethod("colSums",
          signature(x = "gpuMatrix", na.rm = "missing", dims = "missing"),
          function(x, na.rm, dims){
              gpu_colSums(x)
          })


#' @rdname gpuMatrix.colSums
#' @export
setMethod("rowSums",
          signature(x = "gpuMatrix", na.rm = "missing", dims = "missing"),
          function(x, na.rm, dims){
              gpu_rowSums(x)
          })



#' @rdname gpuMatrix.colSums
#' @export
setMethod("colMeans",
          signature(x = "gpuMatrix", na.rm = "missing", dims = "missing"),
          function(x, na.rm, dims){
              gpu_colMeans(x)
          })


#' @rdname gpuMatrix.colSums
#' @export
setMethod("rowMeans",
          signature(x = "gpuMatrix", na.rm = "missing", dims = "missing"),
          function(x, na.rm, dims){
              gpu_rowMeans(x)
          })


#' @title Covariance (gpuMatrix)
#' @param x A gpuMatrix object
#' @param y Not used
#' @param use Not used
#' @param method Character string indicating with covariance to be computed.
#' @return A \code{gpuMatrix} containing the symmetric covariance values.
#' @author Charles Determan Jr.
#' @docType methods
#' @rdname gpuMatrix.cov
#' @aliases cov,gpuMatrix
#' @export
setMethod("cov",
          signature(x = "gpuMatrix", y = "missing", use = "missing", method = "missing"),
          function(x, y = NULL, use = NULL, method = "pearson") {
              if(method != "pearson"){
                  stop("Only pearson covariance implemented")
              }
              return(gpu_pmcc(x))
          })

#' @rdname gpuMatrix.cov
#' @export
setMethod("cov",
          signature(x = "gpuMatrix", y = "missing", use = "missing", method = "character"),
          function(x, y = NULL, use = NULL, method = "pearson") {
              if(method != "pearson"){
                  stop("Only pearson covariance implemented")
              }
              return(gpu_pmcc(x))
          })


#' @title gpuMatrix Crossproduct
#' @description Return the matrix cross-product of two conformable
#' matrices using a GPU.  This is equivalent to t(x) %*% y (crossprod)
#' or x %*% t(t) (tcrossprod) but faster as no data transfer between
#' device and host is required.
#' @param x A gpuMatrix
#' @param y A gpuMatrix
#' @return A gpuMatrix
#' @author Charles Determan Jr.
#' @docType methods
#' @rdname gpuMatrix-crossprod
#' @aliases crossprod,gpuMatrix
#' @export
setMethod("crossprod",
          signature(x = "gpuMatrix", y = "missing"),
          function(x, y){
              gpu_crossprod(x, x)
          })


#' @rdname gpuMatrix-crossprod
#' @export
setMethod("crossprod",
          signature(x = "gpuMatrix", y = "gpuMatrix"),
          function(x, y){
              gpu_crossprod(x, y)
          })

#' @rdname gpuMatrix-crossprod
setMethod("tcrossprod",
          signature(x = "gpuMatrix", y = "missing"),
          function(x, y){
              gpu_tcrossprod(x, x)
          })


#' @rdname gpuMatrix-crossprod
#' @export
setMethod("tcrossprod",
          signature(x = "gpuMatrix", y = "gpuMatrix"),
          function(x, y){
              gpu_tcrossprod(x, y)
          })


#' @rdname dist-vclMatrix
#' @aliases dist,gpuMatrix
#' @export
setMethod("dist", signature(x="gpuMatrix"),
          function(x, method = "euclidean", diag = FALSE, upper = FALSE, p = 2)
          {
              device_flag <- 
                  switch(options("gpuR.default.device")$gpuR.default.device,
                         "cpu" = 1, 
                         "gpu" = 0,
                         stop("unrecognized default device option"
                         )
                  )
              
              type = typeof(x)
              
              if( type == "integer"){
                  stop("Integer type not currently supported")
              }
              
              D <- gpuMatrix(nrow=nrow(x), ncol=nrow(x), type=type)
              
              switch(method,
                     "euclidean" = gpuMatrix_euclidean(
                         x, 
                         D,
                         diag,
                         upper,
                         p),
                     stop("method not currently supported")
              )
              
              return(D)
          }
)


setMethod("deepcopy", signature(object ="gpuMatrix"),
          function(object){
              
              out <- switch(typeof(object),
                            "integer" = new("igpuMatrix",
                                            address = cpp_deepcopy_gpuMatrix(object@address, 4L)),
                            "float" = new("fgpuMatrix", 
                                          address = cpp_deepcopy_gpuMatrix(object@address, 6L)),
                            "double" = new("dgpuMatrix", 
                                           address = cpp_deepcopy_gpuMatrix(object@address, 8L)),
                            stop("unrecognized type")
                            )
              return(out)
              
          })

