# The primary class for all vclMatrix objects

#' @title vclMatrix Class
#' @description This is the 'mother' class for all
#' vclMatrix objects.  These objects are pointers
#' to viennacl matrices directly on the GPU.  This will 
#' avoid the overhead of passing data back and forth 
#' between the host and device.
#' 
#' As such, any changes made
#' to normal R 'copies' (e.g. A <- B) will be propogated to
#' the parent object.
#' 
#' There are multiple child classes that correspond
#' to the particular data type contained.  These include
#' \code{ivclMatrix}, \code{fvclMatrix}, and 
#' \code{dvclMatrix} corresponding to integer, float, and
#' double data types respectively.
#' @section Slots:
#'  Common to all vclMatrix objects in the package
#'  \describe{
#'      \item{\code{address}:}{Pointer to data matrix}
#'      \item{\code{.context_index}:}{Integer index of OpenCL contexts}
#'      \item{\code{.platform_index}:}{Integer index of OpenCL platforms}
#'      \item{\code{.platform}:}{Name of OpenCL platform}
#'      \item{\code{.device_index}:}{Integer index of active device}
#'      \item{\code{.device}:}{Name of active device}
#'  }
#' @note R does not contain a native float type.  As such,
#' the matrix data within a \code{\link{fvclMatrix-class}} 
#' will be represented as double but downcast when any 
#' vclMatrix methods are used.
#' 
#' May also remove the type slot
#' 
#' @name vclMatrix-class
#' @rdname vclMatrix-class
#' @author Charles Determan Jr.
#' @seealso \code{\link{ivclMatrix-class}}, 
#' \code{\link{fvclMatrix-class}},
#' \code{\link{dvclMatrix-class}}
#' @export
setClass('vclMatrix', 
         slots = c(address="externalptr",
                   .context_index = "integer",
                   .platform_index = "integer",
                   .platform = "character",
                   .device_index = "integer",
                   .device = "character"))


#' @title ivclMatrix Class
#' @description An integer type matrix in the S4 \code{vclMatrix}
#' representation.
#' @section Slots:
#'  \describe{
#'      \item{\code{address}:}{Pointer to a integer typed matrix}
#'  }
#' @name ivclMatrix-class
#' @rdname ivclMatrix-class
#' @author Charles Determan Jr.
#' @seealso \code{\link{vclMatrix-class}}, 
#' \code{\link{ivclMatrix-class}},
#' \code{\link{dvclMatrix-class}}
#' @export
setClass("ivclMatrix",
         contains = "vclMatrix",
         validity = function(object) {
             if( typeof(object) != "integer"){
                 return("ivclMatrix must be of type 'integer'")
             }
             TRUE
         })


#' @title fvclMatrix Class
#' @description An integer type matrix in the S4 \code{vclMatrix}
#' representation.
#' @section Slots:
#'  \describe{
#'      \item{\code{address}:}{Pointer to a float matrix.}
#'  }
#' @name fvclMatrix-class
#' @rdname fvclMatrix-class
#' @author Charles Determan Jr.
#' @seealso \code{\link{vclMatrix-class}}, 
#' \code{\link{ivclMatrix-class}},
#' \code{\link{dvclMatrix-class}}
#' @export
setClass("fvclMatrix",
         contains = "vclMatrix",
         validity = function(object) {
             if( typeof(object) != "float"){
                 return("fvclMatrix must be of type 'float'")
             }
             TRUE
         })


#' @title dvclMatrix Class
#' @description An integer type matrix in the S4 \code{vclMatrix}
#' representation.
#' @section Slots:
#'  \describe{
#'      \item{\code{address}:}{Pointer to a double type matrix}
#'  }
#' @name dvclMatrix-class
#' @rdname dvclMatrix-class
#' @author Charles Determan Jr.
#' @seealso \code{\link{vclMatrix-class}}, 
#' \code{\link{ivclMatrix-class}},
#' \code{\link{fvclMatrix-class}}
#' @export
setClass("dvclMatrix",
         contains = "vclMatrix",
         validity = function(object) {
             if( typeof(object) != "double"){
                 return("dvclMatrix must be of type 'double'")
             }
             TRUE
         })


# @export
setClass("ivclMatrixBlock", 
         contains = "ivclMatrix")

# @export
setClass("fvclMatrixBlock", 
         contains = "fvclMatrix")

# @export
setClass("dvclMatrixBlock", 
         contains = "dvclMatrix")

