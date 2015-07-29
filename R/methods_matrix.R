#' @import methods

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
#' @export
setMethod("Arith", c(e1="gpuMatrix", e2="gpuMatrix"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              switch(op,
                     `+` = gpu_Mat_axpy(1, e1, e2),
                     `-` = gpu_Mat_axpy(-1, e2, e1),
                     {
                         stop("undefined operation")
                     }
              )
          },
valueClass = "gpuMatrix"
)


#' @title Get gpuMatrix type
#' @param x A gpuMatrix object
#' @aliases typeof,gpuMatrix
#' @export
setMethod('typeof', signature(x="gpuMatrix"),
          function(x) {
              switch(class(x),
                     "igpuMatrix" = "integer",
                     "fgpuMatrix" = "float",
                     "dgpuMatrix" = "double")
          })


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

#' @title Extract all gpuMatrix elements
#' @param x A gpuMatrix object
#' @param i missing
#' @param j missing
#' @param drop missing
#' @aliases [,gpuMatrix
#' @author Charles Determan Jr.
#' @export
setMethod("[",
          signature(x = "gpuMatrix", i = "missing", j = "missing", drop = "missing"),
          function(x, i, j, drop) {
              switch(typeof(x),
                     "integer" = return(iXptrToSEXP(x@address)),
                     "float" = return(fXptrToSEXP(x@address)),
                     "double" = return(dXptrToSEXP(x@address))
              )
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

