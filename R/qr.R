


# @export
# qr.vclMatrix <- 
#           function(x, ...)
#           {
#               type = typeof(x)
#               
#               if(nrow(x) != ncol(x)){
#                   stop("non-square matrix not currently supported for 'qr'")
#               }
#               
#               if( type == "integer"){
#                   stop("Integer type not currently supported")
#               }
#               
#               Q <- vclMatrix(nrow = nrow(x), ncol = ncol(x), type = type)
#               R <- vclMatrix(nrow = nrow(x), ncol = nrow(x), type = type)
#               
#               switch(type,
#                      "float" = cpp_vclMatrix_qr(x@address, 
#                                                 Q@address,
#                                                 R@address,
#                                                 6L,
#                                                 x@.context_index - 1),
#                      "double" = cpp_vclMatrix_qr(x@address,
#                                                  Q@address,
#                                                  R@address,
#                                                  8L,
#                                                  x@.context_index - 1),
#                      stop("type not currently supported")
#               )
#               
#               return(list(Q = Q, R = R))
#           }
