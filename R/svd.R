
setOldClass("svd")

# GPU Singular Value Decomposition

#' @export
setMethod("svd", 
          signature(x="vclMatrix"),
          function(x){
              
              if(ncol(x) != nrow(x)){
                  stop("non-square matrix not currently supported for 'svd'")
              }
              
              type <- typeof(x)
              
              D <- vclVector(length = as.integer(min(nrow(x), ncol(x))), type = type, ctx_id=x@.context_index)
              U <- vclMatrix(0, ncol = nrow(x), nrow = nrow(x), type = type, ctx_id=x@.context_index)
              V <- vclMatrix(0, ncol = ncol(x), nrow = ncol(x), type = type, ctx_id=x@.context_index)
              
              switch(type,
                     integer = {cpp_vclMatrix_svd(x@address, D@address, U@address, V@address, 4L, ctx_id = x@.context_index - 1)},
                     float = {cpp_vclMatrix_svd(x@address, D@address, U@address, V@address, 6L, ctx_id = x@.context_index - 1)},
                     double = {cpp_vclMatrix_svd(x@address, D@address, U@address, V@address, 8L, ctx_id = x@.context_index - 1)},
                     stop("type not recognized")
              )
              
              return(list(d = D, u = U, v = V))
          }
)

#' @export
setMethod("svd", 
          signature(x="gpuMatrix"),
          function(x){
              
              if(ncol(x) != nrow(x)){
                  stop("non-square matrix not currently supported for 'svd'")
              }
              
              type <- typeof(x)
              
              D <- gpuVector(length = as.integer(min(nrow(x), ncol(x))), type = type, ctx_id=x@.context_index)
              U <- gpuMatrix(0, ncol = nrow(x), nrow = nrow(x), type = type, ctx_id=x@.context_index)
              V <- gpuMatrix(0, ncol = ncol(x), nrow = ncol(x), type = type, ctx_id=x@.context_index)
              
              switch(type,
                     integer = {cpp_gpuMatrix_svd(x@address, D@address, U@address, V@address, 4L, ctx_id = x@.context_index - 1)},
                     float = {cpp_gpuMatrix_svd(x@address, D@address, U@address, V@address, 6L, ctx_id = x@.context_index - 1)},
                     double = {cpp_gpuMatrix_svd(x@address, D@address, U@address, V@address, 8L, ctx_id = x@.context_index - 1)},
                     stop("type not recognized")
              )
              
              return(list(d = D, u = U, v = V))
          }
)
