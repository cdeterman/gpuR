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
              
              if(ncol(a) != ncol(b) || nrow(a) != nrow(b)){
                  stop("matrices must have equivalent dimensions")
              }
              
              type <- typeof(a)
              
              # don't want to overwrite a
              in_mat <- deepcopy(a)
              
              # don't want to overwrite b when passed in
              out_mat <- deepcopy(b)
              
              switch(type,
                     integer = {cpp_vclMatrix_solve(in_mat@address, out_mat@address, 4L)},
                     float = {cpp_vclMatrix_solve(in_mat@address, out_mat@address, 6L)},
                     double = {cpp_vclMatrix_solve(in_mat@address, out_mat@address, 8L)},
                     stop("type not recognized")
              )
              
              return(out_mat)
          }
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
                     integer = {cpp_vclMatrix_solve(in_mat@address, out_mat@address, 4L)},
                     float = {cpp_vclMatrix_solve(in_mat@address, out_mat@address, 6L)},
                     double = {cpp_vclMatrix_solve(in_mat@address, out_mat@address, 8L)},
                     stop("type not recognized")
              )
              
              return(out_mat)
          }
)



