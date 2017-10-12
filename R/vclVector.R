#' @title Construct a vclVector
#' @description Construct a vclVector of a class that inherits
#' from \code{vclVector}.  This class points to memory directly on
#' the GPU to avoid the cost of data transfer between host and device.
#' @param data An object that is or can be converted to a 
#' \code{vector}
#' @param length A non-negative integer specifying the desired length.
#' @param type A character string specifying the type of vclVector.  Default
#' is NULL where type is inherited from the source data type.
#' @param ctx_id An integer specifying the object's context
#' @param ... Additional method to pass to vclVector methods
#' @return A vclVector object
#' @docType methods
#' @rdname vclVector-methods
#' @author Charles Determan Jr.
#' @export
setGeneric("vclVector", function(data, length, type=NULL, ...){
    standardGeneric("vclVector")
})

#' @rdname vclVector-methods
#' @aliases vclVector,vector
setMethod('vclVector', 
          signature(data = 'vector', length = 'missing'),
          function(data, length, type=NULL, ctx_id = NULL){
              
              if (is.null(type)){
                  if(typeof(data) == "integer") {
                      type <- "integer"
                  }else{
                      type <- getOption("gpuR.default.type")    
                  }
              }
              
              if (!missing(length)) {
                  warning("length argument not currently used when passing
                          in data")
              }
              
              device <- if(is.null(ctx_id)) currentDevice() else listContexts()[ctx_id,]
              
              context_index <- ifelse(is.null(ctx_id), currentContext(), ctx_id)
              device_index <- if(is.null(ctx_id)) as.integer(device$device_index) else device$device_index + 1L
              
              platform_index <- if(is.null(ctx_id)) currentPlatform()$platform_index else device$platform_index + 1L
              platform_name <- platformInfo(platform_index)$platformName
              
              device_type <- device$device_type
              device_name <- switch(device_type,
                                    "gpu" = gpuInfo(
                                        device_idx = as.integer(device_index),
                                        context_idx = context_index)$deviceName,
                                    "cpu" = cpuInfo(
                                        device_idx = as.integer(device_index),
                                        context_idx = context_index)$deviceName,
                                    stop("Unrecognized device type")
              )
              
              data = switch(type,
                            integer = {
                                new("ivclVector", 
                                    address=vectorToVCL(data, 4L, context_index - 1),
                                    .context_index = context_index,
                                    .platform_index = platform_index,
                                    .platform = platform_name,
                                    .device_index = device_index,
                                    .device = device_name)
                            },
                            float = {
                                new("fvclVector", 
                                    address=vectorToVCL(data, 6L, context_index - 1),
                                    .context_index = context_index,
                                    .platform_index = platform_index,
                                    .platform = platform_name,
                                    .device_index = device_index,
                                    .device = device_name)
                            },
                            double = {
                                assert_has_double(device_index, context_index)
                                new("dvclVector",
                                    address = vectorToVCL(data, 8L, context_index - 1),
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
          valueClass = "vclVector")


#' @rdname vclVector-methods
#' @aliases vclVector,missing
setMethod('vclVector', 
          signature(data = 'missing'),
          function(data, length, type=NULL, ctx_id=NULL){
              
              if (is.null(type)) type <- getOption("gpuR.default.type")
              if (length <= 0) stop("length must be a positive integer")
              if (!is.integer(length)) stop("length must be a positive integer")
              
              device <- if(is.null(ctx_id)) currentDevice() else listContexts()[ctx_id,]
              
              context_index <- ifelse(is.null(ctx_id), currentContext(), ctx_id)
              device_index <- if(is.null(ctx_id)) as.integer(device$device_index) else device$device_index + 1L
              
              platform_index <- if(is.null(ctx_id)) currentPlatform()$platform_index else device$platform_index + 1L
              platform_name <- platformInfo(platform_index)$platformName
              
              device_type <- device$device_type
              device_name <- switch(device_type,
                                    "gpu" = gpuInfo(
                                        device_idx = as.integer(device_index),
                                        context_idx = context_index)$deviceName,
                                    "cpu" = cpuInfo(
                                        device_idx = as.integer(device_index),
                                        context_idx = context_index)$deviceName,
                                    stop("Unrecognized device type")
              )
              
              data = switch(type,
                            integer = {
                                new("ivclVector", 
                                    address=emptyVecVCL(length, 4L, context_index - 1),
                                    .context_index = context_index,
                                    .platform_index = platform_index,
                                    .platform = platform_name,
                                    .device_index = device_index,
                                    .device = device_name)
                            },
                            float = {
                                new("fvclVector", 
                                    address=emptyVecVCL(length, 6L, context_index - 1),
                                    .context_index = context_index,
                                    .platform_index = platform_index,
                                    .platform = platform_name,
                                    .device_index = device_index,
                                    .device = device_name)
                            },
                            double = {
                                assert_has_double(device_index, context_index)
                                new("dvclVector",
                                    address = emptyVecVCL(length, 8L, context_index - 1),
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
          valueClass = "vclVector"
)



#' @rdname vclVector-methods
#' @aliases vclVector,vector
setMethod('vclVector', 
          signature(data = 'numeric', length = 'numericOrInt'),
          function(data, length, type=NULL, ctx_id = NULL){
              
              if (is.null(type)) type <- getOption("gpuR.default.type")
              
              device <- if(is.null(ctx_id)) currentDevice() else listContexts()[ctx_id,]
              
              context_index <- ifelse(is.null(ctx_id), currentContext(), ctx_id)
              device_index <- if(is.null(ctx_id)) as.integer(device$device_index) else device$device_index + 1L
              
              platform_index <- if(is.null(ctx_id)) currentPlatform()$platform_index else device$platform_index + 1L
              platform_name <- platformInfo(platform_index)$platformName
              
              device_type <- device$device_type
              device_name <- switch(device_type,
                                    "gpu" = gpuInfo(
                                        device_idx = as.integer(device_index),
                                        context_idx = context_index)$deviceName,
                                    "cpu" = cpuInfo(
                                        device_idx = as.integer(device_index),
                                        context_idx = context_index)$deviceName,
                                    stop("Unrecognized device type")
              )
              
              data = switch(type,
                            integer = {
                                new("ivclVector", 
                                    address=cpp_scalar_vclVector(data, length, 4L, context_index - 1),
                                    .context_index = context_index,
                                    .platform_index = platform_index,
                                    .platform = platform_name,
                                    .device_index = device_index,
                                    .device = device_name)
                            },
                            float = {
                                new("fvclVector", 
                                    address=cpp_scalar_vclVector(data, length, 6L, context_index - 1),
                                    .context_index = context_index,
                                    .platform_index = platform_index,
                                    .platform = platform_name,
                                    .device_index = device_index,
                                    .device = device_name)
                            },
                            double = {
                                assert_has_double(device_index, context_index)
                                new("dvclVector",
                                    address = cpp_scalar_vclVector(data, length, 8L, context_index - 1),
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
          valueClass = "vclVector")


#' @rdname vclVector-methods
#' @param col index of column to extract from \code{vclMatrix}
#' @param row index of row to extract from \code{vclMatrix}
#' @aliases vclVector,vector
setMethod('vclVector', 
          signature(data = 'vclMatrix', length = 'missing'),
          function(data, length=NULL, type=NULL, ctx_id = NULL, col = NULL, row = NULL){
              
              # print('called correctly')
              
              if (is.null(type)){
                  type <- typeof(data)  
              }else{
                  if(type != typeof(data)){
                      stop("type must match parent matrix")
                  }
              }
              
              if(!is.null(col) && !is.null(row)){
                  stop("only a single column or row can be extracted")
              }
              if(length(col) > 1 || length(row) > 1){
                  stop("only a single column or row can be extracted")
              }
              
              
              context_index <- data@.context_index
              platform_index <- data@.platform_index
              device_index <- data@.device_index
              device_type <- deviceType(platform_index, device_index)
              device_name <- switch(device_type,
                                    "gpu" = gpuInfo(
                                        device_idx = as.integer(device_index),
                                        context_idx = context_index)$deviceName,
                                    "cpu" = cpuInfo(
                                        device_idx = as.integer(device_index),
                                        context_idx = context_index)$deviceName,
                                    stop("Unrecognized device type")
              )
              platform_name <- platformInfo(platform_index)$platformName
              
              if(!is.null(col)){
                  data = switch(type,
                                integer = {
                                    new("ivclVector", 
                                        address=extractCol(data@address, col - 1, 4L, context_index - 1),
                                        .context_index = context_index,
                                        .platform_index = platform_index,
                                        .platform = platform_name,
                                        .device_index = device_index,
                                        .device = device_name)
                                },
                                float = {
                                    new("fvclVector", 
                                        address=extractCol(data@address, col - 1, 6L, context_index - 1),
                                        .context_index = context_index,
                                        .platform_index = platform_index,
                                        .platform = platform_name,
                                        .device_index = device_index,
                                        .device = device_name)
                                },
                                double = {
                                    new("dvclVector",
                                        address = extractCol(data@address, col - 1, 8L, context_index - 1),
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
                                    new("ivclVector", 
                                        address=extractRow(data@address, row - 1, 4L, context_index - 1),
                                        .context_index = context_index,
                                        .platform_index = platform_index,
                                        .platform = platform_name,
                                        .device_index = device_index,
                                        .device = device_name)
                                },
                                float = {
                                    new("fvclVector", 
                                        address=extractRow(data@address, row - 1, 6L, context_index - 1),
                                        .context_index = context_index,
                                        .platform_index = platform_index,
                                        .platform = platform_name,
                                        .device_index = device_index,
                                        .device = device_name)
                                },
                                double = {
                                    new("dvclVector",
                                        address = extractRow(data@address, row - 1, 8L, context_index - 1),
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
          valueClass = "vclVector")
