### gpuVectorSlice Wrappers ###

# GPU axpy wrapper
gpuSliceVec_axpy <- function(alpha, A, B){
    
    device_flag <- 
        switch(options("gpuR.default.device")$gpuR.default.device,
               "cpu" = 1, 
               "gpu" = 0,
               stop("unrecognized default device option"
               )
        )
    
    if(length(A) != length(B)){
        stop("Lengths of vectors do not match")
    }
    
#     pkg_path <- find.package("gpuR", .libPaths())
#     file <- file.path(pkg_path, "CL", "basic_axpy.cl")
#     
#     if(!file_test("-f", file)){
#         stop("kernel file does not exist")
#     }
#     kernel <- readChar(file, file.info(file)$size)
    
    type <- typeof(A)
    
    Z <- gpuVector(length=length(A), type=type)
    if(!missing(B))
    {
        Z@address <- B@address
    }
    
    switch(type,
#            integer = {cpp_gpuVectorSlice_iaxpy(alpha, 
#                                                A@address,
#                                                Z@address, 
#                                                kernel)
#            },
           float = {cpp_gpuVectorSlice_axpy(alpha, 
                                            A@address, 
                                            Z@address, 
                                            device_flag,
                                            6L)
           },
           double = {cpp_gpuVectorSlice_axpy(alpha, 
                                             A@address,
                                             Z@address,
                                             device_flag,
                                             8L)
           },
           stop("type not recognized")
    )
    
    return(Z)
}

