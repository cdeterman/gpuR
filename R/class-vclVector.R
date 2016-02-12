# The primary class for all vclVector objects

#' @title vclVector Class
#' @description This is the 'mother' class for all
#' vclVector objects.  All other vclVector classes
#' inherit from this class but there are no current
#' circumstances where this class is used directly.
#' 
#' There are multiple child classes that correspond
#' to the particular data type contained.  These include
#' \code{ivclVector}.
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
#' @name vclVector-class
#' @rdname vclVector-class
#' @author Charles Determan Jr.
#' @seealso \code{\link{ivclVector-class}}
#' @export
setClass('vclVector', 
         slots = c(address="externalptr",
                   .context_index = "integer",
                   .platform_index = "integer",
                   .platform = "character",
                   .device_index = "integer",
                   .device = "character"))

# setClass('vclVector',
#          representation("VIRTUAL"),
#          validity = function(object) {
#              if( !length(object@object) > 0 ){
#                  return("vclVector must be a length greater than 0")
#              }
#              TRUE
#          })


#' @title ivclVector Class
#' @description An integer vector in the S4 \code{vclVector}
#' representation.
#' @section Slots:
#'  \describe{
#'      \item{\code{address}:}{An integer vector object}
#'  }
#' @name ivclVector-class
#' @rdname ivclVector-class
#' @author Charles Determan Jr.
#' @seealso \code{\link{vclVector-class}}
#' @export
setClass("ivclVector",
         contains = "vclVector",
         validity = function(object) {
             if( typeof(object) != "integer"){
                 return("ivclVector must be of type 'integer'")
             }
             TRUE
         })

# setClass("ivclVector",
#          slots = c(object = "vector"),
#          contains = "vclVector",
#          validity = function(object) {
#              if( typeof(object) != "integer"){
#                  return("ivclVector must be of type 'integer'")
#              }
#              TRUE
#          })


#' @title fvclVector Class
#' @description An float vector in the S4 \code{vclVector}
#' representation.
#' @section Slots:
#'  \describe{
#'      \item{\code{address}:}{Pointer to a float typed vector}
#'  }
#' @name fvclVector-class
#' @rdname fvclVector-class
#' @author Charles Determan Jr.
#' @seealso \code{\link{vclVector-class}}
#' @export
setClass("fvclVector",
         contains = "vclVector",
         validity = function(object) {
             if( typeof(object) != "float"){
                 return("fvclVector must be of type 'float'")
             }
             TRUE
         })


#' @title dvclVector Class
#' @description An double vector in the S4 \code{vclVector}
#' representation.
#' @section Slots:
#'  \describe{
#'      \item{\code{address}:}{Pointer to a double typed vector}
#'  }
#' @name dvclVector-class
#' @rdname dvclVector-class
#' @author Charles Determan Jr.
#' @seealso \code{\link{vclVector-class}}
#' @export
setClass("dvclVector",
         contains = "vclVector",
         validity = function(object) {
             if( typeof(object) != "double"){
                 return("dvclVector must be of type 'double'")
             }
             TRUE
         })

# @export
setClass("ivclVectorSlice", 
         contains = "ivclVector")

# @export
setClass("fvclVectorSlice", 
         contains = "fvclVector")

# @export
setClass("dvclVectorSlice", 
         contains = "dvclVector")

