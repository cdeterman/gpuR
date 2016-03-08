
# need code to reshape if dimensions differ from input

#' @title Construct a vclMatrix
#' @description Construct a vclMatrix of a class that inherits
#' from \code{vclMatrix}.  This class points to memory directly on
#' the GPU to avoid the cost of data transfer between host and device.
#' @param data An object that is or can be converted to a 
#' \code{matrix} object
#' @param nrow An integer specifying the number of rows
#' @param ncol An integer specifying the number of columns
#' @param type A character string specifying the type of vclMatrix.  Default
#' is NULL where type is inherited from the source data type.
#' @param ... Additional method to pass to vclMatrix methods
#' @return A vclMatrix object
#' @docType methods
#' @rdname vclMatrix-methods
#' @author Charles Determan Jr.
#' @export
setGeneric("vclMatrix", function(data = NA, nrow=NA, ncol=NA, type=NULL, ...){
    standardGeneric("vclMatrix")
})

#' @rdname vclMatrix-methods
#' @aliases vclMatrix,matrix
setMethod('vclMatrix', 
          signature(data = 'matrix'),
          function(data, type=NULL){
              
              if (is.null(type)) type <- typeof(data)
              device_flag <- ifelse(options("gpuR.default.device.type") == "gpu", 0, 1)
              
              device <- currentDevice()
              
              context_index <- currentContext()
              device_index <- device$device_index
              device_type <- device$device_type
              device_name <- switch(device_type,
                                    "gpu" = gpuInfo(device_idx = as.integer(device_index))$deviceName,
                                    "cpu" = cpuInfo(device_idx = as.integer(device_index))$deviceName,
                                    stop("Unrecognized device type")
              )
              platform_index <- currentPlatform()$platform_index
              platform_name <- platformInfo(platform_index)$platformName
              
              if(type == "double" & !deviceHasDouble(platform_index, device_index)){
                  stop("Double precision not supported for current device. 
                       Try setting 'type = 'float'' or change device if multiple available.")
              }
              
              data = switch(type,
                            integer = {
                                new("ivclMatrix", 
                                    address=cpp_sexp_mat_to_vclMatrix(data, 4L, device_flag),
                                    .context_index = context_index,
                                    .platform_index = platform_index,
                                    .platform = platform_name,
                                    .device_index = device_index,
                                    .device = device_name)
                            },
                            float = {
                                new("fvclMatrix", 
                                    address=cpp_sexp_mat_to_vclMatrix(data, 6L, device_flag),
                                    .context_index = context_index,
                                    .platform_index = platform_index,
                                    .platform = platform_name,
                                    .device_index = device_index,
                                    .device = device_name)
                            },
                            double = {
                                new("dvclMatrix",
                                    address = cpp_sexp_mat_to_vclMatrix(data, 8L, device_flag),
                                    .context_index = context_index,
                                    .platform_index = platform_index,
                                    .platform = platform_name,
                                    .device_index = device_index,
                                    .device = device_name)
                            },
                            stop("this is an unrecognized 
                                 or unimplemented data type")
                            )
              
              return(data)
          },
          valueClass = "vclMatrix")


#' @rdname vclMatrix-methods
#' @aliases vclMatrix,missing
setMethod('vclMatrix', 
          signature(data = 'missing'),
          function(data, nrow=NA, ncol=NA, type=NULL){
              
              if (is.null(type)) type <- getOption("gpuR.default.type")
              device_flag <- ifelse(options("gpuR.default.device.type") == "gpu", 0, 1)
              
              device <- currentDevice()
              
              context_index <- currentContext()
              device_index <- device$device_index
              device_type <- device$device_type
              device_name <- switch(device_type,
                                    "gpu" = gpuInfo(device_idx = as.integer(device_index))$deviceName,
                                    "cpu" = cpuInfo(device_idx = as.integer(device_index))$deviceName,
                                    stop("Unrecognized device type")
              )
              platform_index <- currentPlatform()$platform_index
              platform_name <- platformInfo(platform_index)$platformName
              
              if(type == "double" & !deviceHasDouble(platform_index, device_index)){
                  stop("Double precision not supported for current device. 
                       Try setting 'type = 'float'' or change device if multiple available.")
              }
              
              data = switch(type,
                            integer = {
                                new("ivclMatrix", 
                                    address=cpp_zero_vclMatrix(nrow, ncol, 4L, device_flag),
                                    .context_index = context_index,
                                    .platform_index = platform_index,
                                    .platform = platform_name,
                                    .device_index = device_index,
                                    .device = device_name)
                            },
                            float = {
                                new("fvclMatrix", 
                                    address=cpp_zero_vclMatrix(nrow, ncol, 6L, device_flag),
                                    .context_index = context_index,
                                    .platform_index = platform_index,
                                    .platform = platform_name,
                                    .device_index = device_index,
                                    .device = device_name)
                            },
                            double = {
                                new("dvclMatrix",
                                    address = cpp_zero_vclMatrix(nrow, ncol, 8L, device_flag),
                                    .context_index = context_index,
                                    .platform_index = platform_index,
                                    .platform = platform_name,
                                    .device_index = device_index,
                                    .device = device_name)
                            },
                            stop("this is an unrecognized 
                                 or unimplemented data type")
                            )
              
              return(data)
          },
          valueClass = "vclMatrix")



#' @rdname vclMatrix-methods
#' @aliases vclMatrix,vector
#' @aliases vclMatrix,numeric
setMethod('vclMatrix', 
          signature(data = 'numeric'),
          function(data, nrow, ncol, type=NULL){
              
              if (is.null(type)) type <- getOption("gpuR.default.type")
              device_flag <- ifelse(options("gpuR.default.device.type") == "gpu", 0, 1)
              
              if(is.na(nrow)) stop("must indicate number of rows: nrow")
              if(is.na(ncol)) stop("must indicate number of columns: ncol")
              assert_is_numeric(nrow)
              assert_is_numeric(ncol)
              
              if(length(data) == 1){
                  data <- vclMatInitNumScalar(data, nrow, ncol, type, device_flag)
              }else{
                  data <- vclMatInitNumVec(data, nrow, ncol, type, device_flag)
              }
              
              return(data)
          },
          valueClass = "vclMatrix")

#' @rdname vclMatrix-methods
#' @aliases vclMatrix,integer
setMethod('vclMatrix',
          signature(data = 'integer'),
          function(data, nrow, ncol, type=NULL){
              
              if (is.null(type)) type <- "integer"
              device_flag <- ifelse(options("gpuR.default.device.type") == "gpu", 0, 1)
              
              if(is.na(nrow)) stop("must indicate number of rows: nrow")
              if(is.na(ncol)) stop("must indicate number of columns: ncol")
              assert_is_numeric(nrow)
              assert_is_numeric(ncol)
              
              if(length(data) == 1){
                  data <- vclMatInitIntScalar(data, nrow, ncol, type, device_flag)
              }else{
                  data <- vclMatInitIntVec(data, nrow, ncol, type, device_flag)
              }
              
              return(data)
          },
          valueClass = "vclMatrix"
)

