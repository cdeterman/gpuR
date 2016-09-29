setOldClass("solve")

# GPU Singular Value Decomposition

#' @title Solve a System of Equations for gpuR objects
#' @description This funciton solves the equation \code{a \%*\% x = b} for 
#' \code{x}, where \code{b} can be either a vector or a matrix.
#' @param a A gpuR object
#' @param b A gpuR object
#' @param ... further arguments passed to or from other methods
#' @return A gpuR object
#' @author Charles Determan Jr.
#' @rdname solve-methods
#' @export
setMethod("solve", 
          signature(a="vclMatrix", b = "vclMatrix"),
          function(a, b, ...){
              
              if(ncol(a) != nrow(a)){
                  stop("non-square matrix not currently supported for 'solve'")
              }
              
              if(nrow(a) != nrow(b)){
                  adim <- paste0("(", paste0(dim(a), collapse = " x "), ")")
                  bdim <- paste0("(", paste0(dim(b), collapse = " x "), ")")
                  s <- paste0("'b' ", bdim, " must be compatible with 'a' ", adim, sep = "")
                  stop(s)
              }
              
              type <- typeof(a)
              
              # don't want to overwrite a
              in_mat <- deepcopy(a)
              
              # don't want to overwrite b when passed in
              out_mat <- deepcopy(b)
              
              switch(type,
                     integer = {
                         stop("Integer solve not implemented")
                     },
                     float = {
                         cpp_gpuMatrix_solve(in_mat@address, 
                                             out_mat@address, 
                                             TRUE,
                                             TRUE,
                                             6L, 
                                             in_mat@.context_index - 1)
                     },
                     double = {
                         cpp_gpuMatrix_solve(in_mat@address, 
                                             out_mat@address, 
                                             TRUE,
                                             TRUE,
                                             8L, 
                                             in_mat@.context_index - 1)
                     },
                     stop("type not recognized")
              )
              
              return(out_mat)
          },
          valueClass = "vclMatrix"
)


#' @rdname solve-methods
#' @export
setMethod("solve", 
          signature(a="vclMatrix", b = "missing"),
          function(a, b, ...){
              
              if(ncol(a) != nrow(a)){
                  stop("non-square matrix not currently supported for 'solve'")
              }
              
              type <- typeof(a)
              
              # don't want to overwrite a
              in_mat <- deepcopy(a)
              
              # don't want to overwrite b when passed in
              out_mat <- identity_matrix(nrow(in_mat), type = type)
              
              switch(type,
                     integer = {
                         stop("Integer solve not implemented")
                     },
                     float = {
                         cpp_gpuMatrix_solve(in_mat@address, 
                                             out_mat@address, 
                                             TRUE,
                                             TRUE,
                                             6L, 
                                             in_mat@.context_index - 1)
                     },
                     double = {
                         # cpp_vclMatrix_solve(in_mat@address, out_mat@address, 8L)
                         cpp_gpuMatrix_solve(in_mat@address, 
                                             out_mat@address, 
                                             TRUE,
                                             TRUE,
                                             8L, 
                                             in_mat@.context_index - 1)
                         },
                     stop("type not recognized")
              )
              
              return(out_mat)
          },
          valueClass = "vclMatrix"
)


#' @rdname solve-methods
#' @export
setMethod("solve", 
          signature(a="gpuMatrix", b = "gpuMatrix"),
          function(a, b, ...){
              
              if(ncol(a) != nrow(a)){
                  stop("non-square matrix not currently supported for 'solve'")
              }
              
              if(nrow(a) != nrow(b)){
                  adim <- paste0("(", paste0(dim(a), collapse = " x "), ")")
                  bdim <- paste0("(", paste0(dim(b), collapse = " x "), ")")
                  s <- paste0("'b' ", bdim, " must be compatible with 'a' ", adim, sep = "")
                  stop(s)
              }
              
              type <- typeof(a)
              
              # don't want to overwrite a
              in_mat <- deepcopy(a)
              
              # don't want to overwrite b when passed in
              out_mat <- deepcopy(b)
              
              switch(type,
                     integer = {
                         stop("Integer solve not implemented")
                     },
                     float = {
                         cpp_gpuMatrix_solve(in_mat@address, 
                                             out_mat@address, 
                                             FALSE,
                                             FALSE,
                                             6L, 
                                             in_mat@.context_index - 1)
                     },
                     double = {
                         cpp_gpuMatrix_solve(in_mat@address, 
                                             out_mat@address, 
                                             FALSE,
                                             FALSE,
                                             8L, 
                                             in_mat@.context_index - 1)
                     },
                     stop("type not recognized")
              )
              
              return(out_mat)
          },
          valueClass = "gpuMatrix"
)


#' @rdname solve-methods
#' @export
setMethod("solve", 
          signature(a="gpuMatrix", b = "missing"),
          function(a, b, ...){
              
              if(ncol(a) != nrow(a)){
                  stop("non-square matrix not currently supported for 'solve'")
              }
              
              type <- typeof(a)
              
              # don't want to overwrite a
              in_mat <- deepcopy(a)
              
              # don't want to overwrite b when passed in
              out_mat <- identity_matrix(nrow(in_mat), type = type)
              
              switch(type,
                     integer = {
                         stop("Integer solve not implemented")
                     },
                     float = {
                         cpp_gpuMatrix_solve(in_mat@address, 
                                             out_mat@address, 
                                             FALSE,
                                             TRUE,
                                             6L, 
                                             in_mat@.context_index - 1)
                     },
                     double = {
                         cpp_gpuMatrix_solve(in_mat@address, 
                                             out_mat@address, 
                                             FALSE,
                                             TRUE,
                                             8L, 
                                             in_mat@.context_index - 1)
                     },
                     stop("type not recognized")
              )
              
              return(out_mat)
          },
          valueClass = "vclMatrix"
)

