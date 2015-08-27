
#' @title Extract or Replace Parts of vclMatrix
#' @param x A vclMatrix object
#' @param i indices specifying rows
#' @param j indices specifying columns
#' @param drop missing
#' @param value data of similar type to be added to vclMatrix object
#' @aliases [,vclMatrix
#' @aliases [<-,vclMatrix
#' @author Charles Determan Jr.
#' @rdname extract-vclMatrix
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

#' @rdname extract-vclMatrix
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


#' @rdname extract-vclMatrix
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

#' @rdname extract-vclMatrix
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

#' @rdname extract-vclMatrix
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

#' @rdname extract-vclMatrix
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

#' @rdname extract-vclMatrix
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

#' @rdname extract-vclMatrix
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


#' @rdname extract-vclMatrix
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

#' @rdname extract-vclMatrix
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
 
#' @title vclMatrix Multiplication
#' @param x A vclMatrix object
#' @param y A vclMatrix object
#' @return A vclMatrix
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

#' @title vclMatrix Arith methods
#' @param e1 A vclMatrix object
#' @param e2 A vclMatrix object
#' @return A vclMatrix object
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
{
    stop("undefined operation")
}
              )
          },
valueClass = "vclMatrix"
)


#' @title vclMatrix Math methods
#' @param x A vclMatrix object
#' @return A vclMatrix object
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
                     stop("undefined operation")
              )
          },
          valueClass = "vclMatrix"
)

#' @title vclMatrix Logarithms
#' @param x A vclMatrix object
#' @return A vclMatrix object
#' @param base A positive number (complex not currently supported by OpenCL):
#' the base with respect to which logarithms are computed.  Defaults to the
#' natural log.
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



#' @title The Number of Rows/Columns of a vclMatrix
#' @param x A vclMatrix object
#' @return An integer of length 1
#' @rdname nrow.vclMatrix
#' @aliases nrow,vclMatrix
#' @aliases ncol,vclMatrix
#' @author Charles Determan Jr.
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

#' @rdname nrow.vclMatrix
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


#' @title vclMatrix dim method
#' @param x A vclMatrix object
#' @return A length 2 vector of the number of rows and columns respectively.
#' @author Charles Determan Jr.
#' @aliases dim,vclMatrix
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

# #' @title Row and Column Sums and Means of vclMatrix
# #' @description Row and column sums and of vclMatrix objects
# #' @param x A vclMatrix object
# #' @param na.rm Not currently used
# #' @param dims Not currently used
# #' @return A gpuVector object
# #' @author Charles Determan Jr.
# #' @docType methods
# #' @rdname vclMatrix.colSums
# #' @aliases colSums,vclMatrix
# #' @aliases rowSums,vclMatrix
# #' @export
# setMethod("colSums",
#           signature(x = "vclMatrix", na.rm = "missing", dims = "missing"),
#           function(x, na.rm, dims){
#               gpu_colSums(x)
#           })
# 
# 
# #' @rdname vclMatrix.colSums
# #' @export
# setMethod("rowSums",
#           signature(x = "vclMatrix", na.rm = "missing", dims = "missing"),
#           function(x, na.rm, dims){
#               gpu_rowSums(x)
#           })
# 
# 
# 
# #' @rdname vclMatrix.colSums
# #' @export
# setMethod("colMeans",
#           signature(x = "vclMatrix", na.rm = "missing", dims = "missing"),
#           function(x, na.rm, dims){
#               gpu_colMeans(x)
#           })
# 
# 
# #' @rdname vclMatrix.colSums
# #' @export
# setMethod("rowMeans",
#           signature(x = "vclMatrix", na.rm = "missing", dims = "missing"),
#           function(x, na.rm, dims){
#               gpu_rowMeans(x)
#           })
# 
# 
# #' @title Covariance (vclMatrix)
# #' @param x A vclMatrix object
# #' @param y Not used
# #' @param use Not used
# #' @param method Character string indicating with covariance to be computed.
# #' @return A \code{vclMatrix} containing the symmetric covariance values.
# #' @author Charles Determan Jr.
# #' @docType methods
# #' @rdname vclMatrix.cov
# #' @aliases cov,vclMatrix
# #' @export
# setMethod("cov",
#           signature(x = "vclMatrix", y = "missing", use = "missing", method = "missing"),
#           function(x, y = NULL, use = NULL, method = "pearson") {
#               if(method != "pearson"){
#                   stop("Only pearson covariance implemented")
#               }
#               return(gpu_pmcc(x))
#           })
# 
# #' @rdname vclMatrix.cov
# #' @export
# setMethod("cov",
#           signature(x = "vclMatrix", y = "missing", use = "missing", method = "character"),
#           function(x, y = NULL, use = NULL, method = "pearson") {
#               if(method != "pearson"){
#                   stop("Only pearson covariance implemented")
#               }
#               return(gpu_pmcc(x))
#           })

