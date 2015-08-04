
# vclMatrix GEMM
vclMatMult <- function(A, B){
    
#     pkg_path <- find.package("gpuR", .libPaths())
#     file <- file.path(pkg_path, "CL", "basic_gemm.cl")
#     
#     if(!file_test("-f", file)){
#         stop("kernel file does not exist")
#     }
#     kernel <- readChar(file, file.info(file)$size)
    
    type <- typeof(A)
    
    C <- vclMatrix(nrow=nrow(A), ncol=ncol(B), type=type)
    
    switch(type,
           integer = {
               stop("OpenCL integer GEMM not currently
                    supported for viennacl matrices")
                #cpp_vclMatrix_igemm(A@address,
                #                   B@address, 
                #                   C@address,
                #                   kernel)
               #                      cpp_vienna_vclMatrix_igemm(A@address,
               #                                                        B@address,
               #                                                        C@address)
           },
           float = {cpp_vienna_vclMatrix_sgemm(A@address,
                                               B@address,
                                               C@address)
           },
           double = {
               if(!deviceHasDouble()){
                   stop("Selected GPU does not support double precision")
               }else{cpp_vienna_vclMatrix_dgemm(A@address,
                                                B@address,
                                                C@address)
               }
           },
{
    stop("type not recognized")
})
return(C)
}

# vclMatrix AXPY
vclMat_axpy <- function(alpha, A, B){
    
    nrA = nrow(A)
    ncA = ncol(A)
    nrB = nrow(B)
    ncB = ncol(B)
    
#     pkg_path <- find.package("gpuR", .libPaths())
#     file <- file.path(pkg_path, "CL", "basic_axpy.cl")
#     
#     if(!file_test("-f", file)){
#         stop("kernel file does not exist")
#     }
#     kernel <- readChar(file, file.info(file)$size)
    
    type <- typeof(A)
    
    Z <- vclMatrix(nrow=nrB, ncol=ncA, type=type)
    if(!missing(B))
    {
        if(length(B[]) != length(A[])) stop("Lengths of matrices must match")
        Z@address <- B@address
    }
    
    switch(type,
           integer = {
               stop("OpenCL integer GEMM not currently
                    supported for viennacl matrices")
               #cpp_gpuMatrix_iaxpy(alpha, 
                #                          A@address,
                #                          Z@address, 
                #                          kernel)
           },
           float = {cpp_vclMatrix_saxpy(alpha, 
                                               A@address, 
                                               Z@address)
           },
           double = {cpp_vclMatrix_daxpy(alpha, 
                                                A@address,
                                                Z@address)
           },
{
    stop("type not recognized")
}
    )

return(Z)
}

# vclMatrix crossprod
vcl_crossprod <- function(X, Y){
    
    if(ncol(X) != ncol(Y)){
        stop("matrices non-conformable")
    }
    
    type <- typeof(X)
    
    Z <- vclMatrix(nrow = ncol(X), ncol = ncol(Y), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vclMatrix_scrossprod(X@address, Y@address, Z@address),
           "double" = cpp_vclMatrix_dcrossprod(X@address, Y@address, Z@address)
    )
    
    return(Z)
}

# vclMatrix crossprod
vcl_tcrossprod <- function(X, Y){
    
    if(nrow(X) != nrow(Y)){
        stop("matrices non-conformable")
    }
    
    type <- typeof(X)
    
    Z <- vclMatrix(nrow = nrow(X), ncol = nrow(Y), type = type)
    
    switch(type,
           "integer" = stop("integer type not currently implemented"),
           "float" = cpp_vclMatrix_stcrossprod(X@address, Y@address, Z@address),
           "double" = cpp_vclMatrix_dtcrossprod(X@address, Y@address, Z@address)
    )
    
    return(Z)
}
