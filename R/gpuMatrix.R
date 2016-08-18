
# need code to reshape if dimensions differ from input

#' @title Construct a gpuMatrix
#' @description Construct a gpuMatrix of a class that inherits
#' from \code{gpuMatrix}
#' @param data An object that is or can be converted to a 
#' \code{matrix} object
#' @param nrow An integer specifying the number of rows
#' @param ncol An integer specifying the number of columns
#' @param type A character string specifying the type of gpuMatrix.  Default
#' is NULL where type is inherited from the source data type.
#' @param ctx_id An integer specifying the object's context
#' @param ... Additional method to pass to gpuMatrix methods
#' @return A gpuMatrix object
#' @docType methods
#' @rdname gpuMatrix-methods
#' @author Charles Determan Jr.
#' @export
setGeneric("gpuMatrix", function(data = NA, nrow=NA, ncol=NA, type=NULL, ...){
    standardGeneric("gpuMatrix")
})

#' @rdname gpuMatrix-methods
#' @aliases gpuMatrix,matrix
setMethod('gpuMatrix', 
          signature(data = 'matrix'),
          function(data, type=NULL, ctx_id = NULL){
              
              if (is.null(type)) type <- typeof(data)
              
              device <- currentDevice()
              
              context_index <- ifelse(is.null(ctx_id), currentContext(), as.integer(ctx_id))
              device_index <- as.integer(device$device_index)
              device_type <- device$device_type
              device_name <- switch(device_type,
                                    "gpu" = gpuInfo(device_idx = as.integer(device_index))$deviceName,
                                    "cpu" = cpuInfo(device_idx = as.integer(device_index))$deviceName,
                                    stop("Unrecognized device type")
              )
              platform_index <- currentPlatform()$platform_index
              platform_name <- platformInfo(platform_index)$platformName
              
              data = switch(type,
                            integer = {
                                new("igpuMatrix", 
                                    address=getRmatEigenAddress(data, 
                                                            nrow(data),
                                                            ncol(data), 
                                                            4L),
                                    .context_index = context_index,
                                    .platform_index = platform_index,
                                    .platform = platform_name,
                                    .device_index = device_index,
                                    .device = device_name)
                            },
                            float = {
                                new("fgpuMatrix", 
                                    address=getRmatEigenAddress(data, 
                                                            nrow(data),
                                                            ncol(data), 
                                                            6L),
                                    .context_index = context_index,
                                    .platform_index = platform_index,
                                    .platform = platform_name,
                                    .device_index = device_index,
                                    .device = device_name)
                            },
                            double = {
                                assert_has_double(platform_index, device_index)
                                new("dgpuMatrix",
                                    address = getRmatEigenAddress(data, 
                                                              nrow(data),
                                                              ncol(data), 
                                                              8L),
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
          valueClass = "gpuMatrix")


#' @rdname gpuMatrix-methods
#' @aliases gpuMatrix,missing
setMethod('gpuMatrix', 
          signature(data = 'missing'),
          function(data, nrow=NA, ncol=NA, type=NULL, ctx_id = NULL){
              
              if (is.null(type)) type <- getOption("gpuR.default.type")
              
              assert_is_numeric(nrow)
              assert_is_numeric(ncol)
              
              device <- currentDevice()
              
              context_index <- ifelse(is.null(ctx_id), currentContext(), as.integer(ctx_id))
              device_index <- as.integer(device$device_index)
              device_type <- device$device_type
              device_name <- switch(device_type,
                                    "gpu" = gpuInfo(device_idx = as.integer(device_index))$deviceName,
                                    "cpu" = cpuInfo(device_idx = as.integer(device_index))$deviceName,
                                    stop("Unrecognized device type")
              )
              platform_index <- currentPlatform()$platform_index
              platform_name <- platformInfo(platform_index)$platformName
              
              data = switch(type,
                            integer = {
                                new("igpuMatrix", 
                                    address=emptyEigenXptr(nrow, ncol, 4L),
                                    .context_index = context_index,
                                    .platform_index = platform_index,
                                    .platform = platform_name,
                                    .device_index = device_index,
                                    .device = device_name)
                            },
                            float = {
                                new("fgpuMatrix", 
                                    address=emptyEigenXptr(nrow, ncol, 6L),
                                    .context_index = context_index,
                                    .platform_index = platform_index,
                                    .platform = platform_name,
                                    .device_index = device_index,
                                    .device = device_name)
                            },
                            double = {
                                assert_has_double(platform_index, device_index)
                                new("dgpuMatrix",
                                    address = emptyEigenXptr(nrow, ncol, 8L),
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
          valueClass = "gpuMatrix")



#' @rdname gpuMatrix-methods
#' @aliases gpuMatrix,numeric
setMethod('gpuMatrix', 
          signature(data = 'numeric'),
          function(data, nrow, ncol, type=NULL, ctx_id = NULL){
              
              if (is.null(type)) type <- "double"
                            
              assert_is_numeric(nrow)
              assert_is_numeric(ncol)
              
              device <- currentDevice()
              
              context_index <- ifelse(is.null(ctx_id), currentContext(), as.integer(ctx_id))
              device_index <- as.integer(device$device_index)
              device_type <- device$device_type
              device_name <- switch(device_type,
                                    "gpu" = gpuInfo(device_idx = as.integer(device_index))$deviceName,
                                    "cpu" = cpuInfo(device_idx = as.integer(device_index))$deviceName,
                                    stop("Unrecognized device type")
              )
              platform_index <- currentPlatform()$platform_index
              platform_name <- platformInfo(platform_index)$platformName
              
              if(length(data) > 1){
                  data = switch(type,
                                integer = stop("Cannot create integer gpuMatrix from numeric"),
                                float = {
                                    new("fgpuMatrix", 
                                        address=sexpVecToEigenXptr(data, nrow, ncol, 6L),
                                        .context_index = context_index,
                                        .platform_index = platform_index,
                                        .platform = platform_name,
                                        .device_index = device_index,
                                        .device = device_name)
                                },
                                double = {
                                    assert_has_double(platform_index, device_index)
                                    new("dgpuMatrix",
                                        address = sexpVecToEigenXptr(data, nrow, ncol, 8L),
                                        .context_index = context_index,
                                        .platform_index = platform_index,
                                        .platform = platform_name,
                                        .device_index = device_index,
                                        .device = device_name)
                                },
                                stop("this is an unrecognized 
                                 or unimplemented data type")
                  )
              }else{
                  data = switch(type,
                                integer = stop("Cannot create integer gpuMatrix from numeric"),
                                float = {
                                    new("fgpuMatrix", 
                                        address=initScalarEigenXptr(data, nrow, ncol, 6L),
                                        .context_index = context_index,
                                        .platform_index = platform_index,
                                        .platform = platform_name,
                                        .device_index = device_index,
                                        .device = device_name)
                                },
                                double = {
                                    assert_has_double(platform_index, device_index)
                                    new("dgpuMatrix",
                                        address = initScalarEigenXptr(data, nrow, ncol, 8L),
                                        .context_index = context_index,
                                        .platform_index = platform_index,
                                        .platform = platform_name,
                                        .device_index = device_index,
                                        .device = device_name)
                                },
                                stop("this is an unrecognized 
                                 or unimplemented data type")
                  )
              }
              
              return(data)
          },
          valueClass = "gpuMatrix")


#' @rdname gpuMatrix-methods
#' @aliases gpuMatrix,integer
setMethod('gpuMatrix', 
          signature(data = 'integer'),
          function(data, nrow, ncol, type=NULL, ctx_id = NULL){
              
              if (is.null(type)) type <- "integer"
              
              assert_is_numeric(nrow)
              assert_is_numeric(ncol)
              
              device <- currentDevice()
              
              context_index <- ifelse(is.null(ctx_id), currentContext(), as.integer(ctx_id))
              device_index <- as.integer(device$device_index)
              device_type <- device$device_type
              device_name <- switch(device_type,
                                    "gpu" = gpuInfo(device_idx = as.integer(device_index))$deviceName,
                                    "cpu" = cpuInfo(device_idx = as.integer(device_index))$deviceName,
                                    stop("Unrecognized device type")
              )
              platform_index <- currentPlatform()$platform_index
              platform_name <- platformInfo(platform_index)$platformName
              
              if(length(data) > 1){
                  data = switch(type,
                                integer = {
                                    new("igpuMatrix", 
                                        address=sexpVecToEigenXptr(data, nrow, ncol, 4L),
                                        .context_index = context_index,
                                        .platform_index = platform_index,
                                        .platform = platform_name,
                                        .device_index = device_index,
                                        .device = device_name)
                                },
                                float = {
                                    new("fgpuMatrix", 
                                        address=sexpVecToEigenXptr(data, nrow, ncol, 6L),
                                        .context_index = context_index,
                                        .platform_index = platform_index,
                                        .platform = platform_name,
                                        .device_index = device_index,
                                        .device = device_name)
                                },
                                double = {
                                    assert_has_double(platform_index, device_index)
                                    new("dgpuMatrix",
                                        address = sexpVecToEigenXptr(data, nrow, ncol, 8L),
                                        .context_index = context_index,
                                        .platform_index = platform_index,
                                        .platform = platform_name,
                                        .device_index = device_index,
                                        .device = device_name)
                                },
                                stop("this is an unrecognized 
                                 or unimplemented data type")
                  )
              }else{
                  data = switch(type,
                                integer = {
                                    new("igpuMatrix", 
                                        address=initScalarEigenXptr(data, nrow, ncol, 4L),
                                        .context_index = context_index,
                                        .platform_index = platform_index,
                                        .platform = platform_name,
                                        .device_index = device_index,
                                        .device = device_name)
                                },
                                float = {
                                    new("fgpuMatrix", 
                                        address=initScalarEigenXptr(data, nrow, ncol, 6L),
                                        .context_index = context_index,
                                        .platform_index = platform_index,
                                        .platform = platform_name,
                                        .device_index = device_index,
                                        .device = device_name)
                                },
                                double = {
                                    assert_has_double(platform_index, device_index)
                                    new("dgpuMatrix",
                                        address = initScalarEigenXptr(data, nrow, ncol, 8L),
                                        .context_index = context_index,
                                        .platform_index = platform_index,
                                        .platform = platform_name,
                                        .device_index = device_index,
                                        .device = device_name)
                                },
                                stop("this is an unrecognized 
                                 or unimplemented data type")
                  )
              }
              
              return(data)
          },
          valueClass = "gpuMatrix")
