

# GPU Element-Wise Absolute Value
vclVecElemMaxAbs <- function(A){
    
    type <- typeof(A)
    
    out <- switch(type,
                  integer = {
                      stop("integer not currently implemented")
                  },
                  float = {cpp_vclVector_elem_max_abs(A@address,
                                                      6L)
                  },
                  double = {
                      cpp_vclVector_elem_max_abs(A@address,
                                                 8L)
                  },
                  stop("type not recognized")
    )
    return(out)
}

# GPU Vector maximum
vclVecMax <- function(A){
    
    type <- typeof(A)
    
    C <- switch(type,
                integer = {
                    stop("integer not currently implemented")
                },
                float = {cpp_vclVector_max(A@address, 6L)
                },
                double = {
                    cpp_vclVector_max(A@address, 8L)
                },
                stop("type not recognized")
    )
    return(C)
}


# GPU Vector minimum
vclVecMin <- function(A){
    
    type <- typeof(A)
    
    C <- switch(type,
                integer = {
                    stop("integer not currently implemented")
                },
                float = {cpp_vclVector_min(A@address, 6L)
                },
                double = {
                    cpp_vclVector_min(A@address, 8L)
                },
                stop("type not recognized")
    )
    return(C)
}


