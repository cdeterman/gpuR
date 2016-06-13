

#' @export
qr_test <- 
          function(x, ...)
          {
              type = typeof(x)
              
              if( type == "integer"){
                  stop("Integer type not currently supported")
              }
              
              switch(type,
                     "float" = cpp_vclMatrix_qr(x@address, 
                                             6L,
                                             x@.context_index - 1),
                     "double" = cpp_vclMatrix_qr(x@address,
                                              8L,
                                              x@.context_index - 1),
                     stop("type not currently supported")
              )
          }
