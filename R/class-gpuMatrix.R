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
#'      \item{\code{.context_index}:}{Integer index of OpenCL contexts}
#'      \item{\code{.platform_index}:}{Integer index of OpenCL platforms}
#'      \item{\code{.platform}:}{Name of OpenCL platform}
#'      \item{\code{.device_index}:}{Integer index of active device}
#'      \item{\code{.device}:}{Name of active device}
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
         slots = c(address="externalptr",
                   .context_index = "integer",
                   .platform_index = "integer",
                   .platform = "character",
                   .device_index = "integer",
                   .device = "character"))


#' @title igpuMatrix Class
#' @description An integer type matrix in the S4 \code{gpuMatrix}
#' representation.
#' @section Slots:
#'  \describe{
#'      \item{\code{address}:}{Pointer to a integer typed matrix}
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


# @export
setClass("igpuMatrixBlock", 
         contains = "igpuMatrix")

# @export
setClass("fgpuMatrixBlock", 
         contains = "fgpuMatrix")

# @export
setClass("dgpuMatrixBlock", 
         contains = "dgpuMatrix")

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
