
#' @export
gpuVector <- function(data = NA, length=1, type='integer'){
    if(is(data, "vector")){
        data = switch(typeof(data),
                      integer = {
                          new("igpuVector", object=data)
                          },
                      stop("unrecognized data type")
                      )
    }
    return(data)
}