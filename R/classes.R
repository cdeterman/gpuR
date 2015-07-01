setClassUnion("missingOrNULL", c("missing", "NULL"))

# The primary class for all gpuVector objects

#' @title gpuVector Class
#' @description This is the 'mother' class for all
#' gpuVector objects.  All other gpuVector classes
#' inherit from this class but there are no current
#' circumstances where this class is used directly.
#' 
#' There are multiple child classes that correspond
#' to the particular data type contained.  These include
#' \code{igpuVector}.
#' @name gpuVector-class
#' @rdname gpuVector-class
#' @author Charles Determan Jr.
#' @seealso \code{\link{igpuVector-class}}
#' @export
setClass('gpuVector',
         representation("VIRTUAL"),
         validity = function(object) {
             if( !length(object@object) > 0 ){
                 return("gpuVector must be a length greater than 0")
             }
             TRUE
         })


#' @title igpuVector Class
#' @description An integer vector in the S4 \code{gpuVector}
#' representation.
#' @section Slots:
#'  \describe{
#'      \item{\code{object}:}{An integer vector object}
#'  }
#' @name igpuVector-class
#' @rdname igpuVector-class
#' @author Charles Determan Jr.
#' @seealso \code{\link{gpuVector-class}}
#' @export
setClass("igpuVector",
         slots = c(object = "vector"),
         contains = "gpuVector",
         validity = function(object) {
             if( typeof(object@object) != "integer"){
                 return("igpuVector must be of type 'integer'")
             }
             TRUE
         })


# The primary class for all gpuMatrix objects

#' @title gpuMatrix Class
#' @description This is the 'mother' class for all
#' gpuMatrix objects.  It is essentially a wrapper for
#' a basic R matrix (possibly to be improved).  All other 
#' gpuMatrix classes inherit from this class but 
#' there are no current circumstances where this class 
#' is used directly.
#' 
#' There are multiple child classes that correspond
#' to the particular data type contained.  These include
#' \code{igpuMatrix}, \code{fgpuMatrix}, and 
#' \code{dgpuMatrix} corresponding to integer, float, and
#' double data types respectively.
#' @section Slots:
#'  Common to all gpuMatrix objects in the package
#'  \describe{
#'      \item{\code{address}:}{Pointer to data matrix}
#'      \item{\code{type}:}{Character object specifying
#'      the type the matrix data will be interpreted as}
#'  }
#' @note R does not contain a native float type.  As such,
#' the matrix data within a \code{\link{fgpuMatrix-class}} 
#' will be represented as double but downcast when any 
#' gpuMatrix methods are used.
#' 
#' May also remove the type slot
#' 
#' @name gpuMatrix-class
#' @rdname gpuMatrix-class
#' @author Charles Determan Jr.
#' @seealso \code{\link{igpuMatrix-class}}, 
#' \code{\link{fgpuMatrix-class}},
#' \code{\link{dgpuMatrix-class}}
#' @export
setClass('gpuMatrix', 
         slots = c(address="externalptr", type="character"))


#' @title igpuMatrix Class
#' @description An integer type matrix in the S4 \code{gpuMatrix}
#' representation.
#' @section Slots:
#'  \describe{
#'      \item{\code{address}:}{Pointer to a integer typed matrix}
#'      \item{\code{type}:}{Character object specifying
#'      the type the matrix data is integer}
#'  }
#' @name igpuMatrix-class
#' @rdname igpuMatrix-class
#' @author Charles Determan Jr.
#' @seealso \code{\link{gpuMatrix-class}}, 
#' \code{\link{igpuMatrix-class}},
#' \code{\link{dgpuMatrix-class}}
#' @export
setClass("igpuMatrix",
         contains = "gpuMatrix",
         validity = function(object) {
             if( typeof(object) != "integer"){
                 return("igpuMatrix must be of type 'integer'")
             }
             TRUE
         })


#' @title fgpuMatrix Class
#' @description An integer type matrix in the S4 \code{gpuMatrix}
#' representation.
#' @section Slots:
#'  \describe{
#'      \item{\code{address}:}{Pointer to a float matrix.}
#'      \item{\code{type}:}{Character object specifying
#'      the type the matrix data is intepreted as float}
#'  }
#' @name fgpuMatrix-class
#' @rdname fgpuMatrix-class
#' @author Charles Determan Jr.
#' @seealso \code{\link{gpuMatrix-class}}, 
#' \code{\link{igpuMatrix-class}},
#' \code{\link{dgpuMatrix-class}}
#' @export
setClass("fgpuMatrix",
         contains = "gpuMatrix",
         validity = function(object) {
             if( typeof(object) != "float"){
                 return("fgpuMatrix must be of type 'float'")
             }
             TRUE
         })


#' @title dgpuMatrix Class
#' @description An integer type matrix in the S4 \code{gpuMatrix}
#' representation.
#' @section Slots:
#'  \describe{
#'      \item{\code{address}:}{Pointer to a double type matrix}
#'      \item{\code{type}:}{Character object specifying
#'      the type the matrix data is double}
#'  }
#' @name dgpuMatrix-class
#' @rdname dgpuMatrix-class
#' @author Charles Determan Jr.
#' @seealso \code{\link{gpuMatrix-class}}, 
#' \code{\link{igpuMatrix-class}},
#' \code{\link{fgpuMatrix-class}}
#' @export
setClass("dgpuMatrix",
         contains = "gpuMatrix",
         validity = function(object) {
             if( typeof(object) != "double"){
                 return("dgpuMatrix must be of type 'double'")
             }
             TRUE
         })

# #' @export
# setClass('gpuMatrix',
#          representation(Dim = "integer", Dimnames = "list", "VIRTUAL"),
#          prototype(Dim = 2L, DImnames = list(NULL, NULL)),
#          validity = function(object) {
#              Dim <- object@Dim
#              if (length(Dim) != 2){
#                  return("Dim slot must be of length 2")
#              }
#              if(any(Dim< 0)){
#                  return("Dim slot must contain non-negative values")
#              }
#              
#              Dimnames <- object@Dimnames
#              if (!is.list(Dimnames) || length(Dimnames) != 2){
#                  return("Dimnames slot must be of length 2")
#              }
#              
#              lenDims <- sapply(Dimnames, length)
#              if (lenDims[1] > 0 && lenDims[1] != Dim[1]){
#                  return("'length(Dimnames[[1]])' must equal Dim[1]")
#              }
#              if (lenDims[2] > 0 && lenDims[2] != Dim[2]){
#                  return("'length(Dimnames[[2]])' must equal Dim[2]")
#              }
#              
#              if( !length(object) > 0 ){
#                  return("gpuMatrix must be a length greater than 0")
#              }
#              TRUE
#          })

