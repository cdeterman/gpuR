


setMethod('as.gpuVector', 
          signature(object = 'vector'),
          function(object, type=NULL){
              if(!typeof(object) %in% c('integer', 'double')){
                  stop("unrecognized data type")
              }
              
              gpuVector(object)
          })

#' @export
setMethod("Arith", c(e1="igpuVector", e2="igpuVector"),
          function(e1, e2)
          {
              op = .Generic[[1]]
              switch(op,
                     `+` = gpu_vec_add(e1@object, e2@object),
                     stop("Undefined operation")
                     )
          }
)
