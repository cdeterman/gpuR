
# need code to reshape if dimensions differ from input

#' @export
setGeneric("gpuMatrix", function(data = NA, ncol=NA, nrow=NA, type=NULL, ...){
    standardGeneric("gpuMatrix")
})

#' @import bigmemory
setMethod('gpuMatrix', 
          signature(data = 'matrix'),
          function(data, ncol=NA, nrow=NA, type=NULL){
              
              if(!is.na(ncol) | !is.na(nrow)){
                  dm <- dim(data)
                  
                  if(is.na(nrow)) nr <- nrow(data) else nr <- nrow
                  if(is.na(ncol)) nc <- ncol(data) else nc <- ncol
                  
                  if(dim[1] != nr | dim[2] != nc){
                      data <- matrix(as.numeric(data), nrow=nr, ncol=nc)
                  }else{
                      data <- matrix(as.numeric(data), nrow=nr, ncol=nc, dimnames=dimnames(data))
                  }
              }
             
              if (is.null(type)) type <- typeof(data)
              
              data = switch(type,
                            integer = {
                                new("igpuMatrix", 
                                    address=as.big.matrix(data, type=type)@address
                                )
                            },
                            float = {
                                new("fgpuMatrix", 
                                    address=as.big.matrix(data, type=type)@address
                                )
                            },
                            double = {
                                new("dgpuMatrix",
                                    address = as.big.matrix(data, type=type)@address
                                )
                            },
                            stop("this is an unrecognized 
                                 or unimplemented data type")
              )
              
              return(data)
          },
          valueClass = "gpuMatrix")


# setMethod('gpuMatrix', 
#           signature(data = 'integer'),
#           function(data, ncol=NA, nrow=NA, type=NULL){
#               
#               if(!is.na(ncol) | !is.na(nrow)){
#                   dm <- dim(data)
#                   
#                   if(is.na(nrow)) nr <- nrow(data) else nr <- nrow
#                   if(is.na(ncol)) nc <- ncol(data) else nc <- ncol
#                   
#                   if(dim[1] != nr | dim[2] != nc){
#                       data <- matrix(as.numeric(data), nrow=nr, ncol=nc)
#                   }else{
#                       data <- matrix(as.numeric(data), nrow=nr, ncol=nc, dimnames=dimnames(data))
#                   }
#               }
#               
#               if (is.null(type)) type <- typeof(data)
#               
#               data = switch(typeof(data),
#                             integer = {
#                                 new("igpuMatrix", 
#                                     address=as.big.matrix(data, type=type)@address
#                                 )
#                             },
#                             stop("this is an unrecognized 
#                                  or unimplemented data type")
#               )
#               
#               return(data)
#           },
#           valueClass = "gpuMatrix")

# GetElements.gpubm <- function(x, i, j, drop=TRUE)
# {
#     if (!is.numeric(i) & !is.character(i) & !is.logical(i))
#         stop("row indices must be numeric, logical, or character vectors.")
#     if (!is.numeric(j) & !is.character(j) & !is.logical(j))
#         stop("column indices must be numeric, logical, or character vectors.")
#     if (is.character(i))
#         if (is.null(rownames(x))) stop("row names do not exist.")
#     else i <- mmap(i, rownames(x))
#     if (is.character(j))
#         if (is.null(colnames(x))) stop("column names do not exist.")
#     else j <- mmap(j, colnames(x))
#     if (is.logical(i)) {
#         if (length(i) != nrow(x))
#             stop("row vector length must match the number of rows of the matrix.")
#         i <- which(i)
#     }
#     if (is.logical(j)) {
#         if (length(j) != ncol(x))
#             stop(paste("column vector length must match the number of",
#                        "columns of the matrix."))
#         j <- which(j)
#     }
#     
#     tempi <- CCleanIndices(as.double(i), as.double(nrow(x)))
#     if (is.null(tempi[[1]])) stop("Illegal row index usage in extraction.\n")
#     if (tempi[[1]]) i <- tempi[[2]]
#     tempj <- CCleanIndices(as.double(j), as.double(ncol(x)))
#     if (is.null(tempj[[1]])) stop("Illegal column index usage in extraction.\n")
#     if (tempj[[1]]) j <- tempj[[2]]
#     
#     retList <- GetMatrixElements(x@address, as.double(j), as.double(i))
#     mat = .addDimnames(retList, length(i), length(j), drop)
#     return(mat)
# }

# Function contributed by Peter Haverty at Genentech.
# GetIndivElements.gpubm <- function(x,i) {
#     # Check i
#     if (is.logical(i)) {
#         stop("Logical indices not allowed when subsetting by a matrix.")
#     }
#     if (ncol(i) != 2) {
#         stop("When subsetting with a matrix, it must have two columns.")
#     }
#     if (is.character(i)) {
#         if (is.null(rownames(x))) stop("row names do not exist.")
#         if (is.null(colnames(x))) stop("column names do not exist.")
#         i <- matrix(c(mmap(i[,1], rownames(x)), mmap(i[,2], colnames(x))), ncol=2)
#     }
#     tempi <- CCleanIndices(as.double(i[,1]), as.double(nrow(x)))
#     if (is.null(tempi[[1]])) stop("Illegal row index usage in assignment.\n")
#     if (tempi[[1]]) i[,1] <- tempi[[2]]
#     tempj <- CCleanIndices(as.double(i[,2]), as.double(ncol(x)))
#     if (is.null(tempj[[1]])) stop("Illegal column index usage in assignment.\n")
#     if (tempj[[1]]) i[,2] <- tempj[[2]]
#     
#     # Call .Call C++
#     return(GetIndivMatrixElements(x@address, as.double(i[,2]),
#                  as.double(i[,1])))
# }

# 
# GetCols.gpubm <- function(x, j, drop=TRUE)
# {
#     if (!is.numeric(j) & !is.character(j) & !is.logical(j))
#         stop("column indices must be numeric, logical, or character vectors.")
#     if (is.character(j))
#         if (is.null(colnames(x))) stop("column names do not exist.")
#     else j <- mmap(j, colnames(x))
#     if (is.logical(j)) {
#         if (length(j) != ncol(x))
#             stop(paste("column vector length must match the number of",
#                        "columns of the matrix."))
#         j <- which(j)
#     }
#     
#     tempj <- CCleanIndices(as.double(j), as.double(ncol(x)))
#     if (is.null(tempj[[1]])) stop("Illegal column index usage in extraction.\n")
#     if (tempj[[1]]) j <- tempj[[2]]
#     
#     retList <- GetMatrixCols(x@address, as.double(j))
#     mat = .addDimnames(retList, nrow(x), length(j), drop)
#     return(mat)
# }
# 
# GetRows.gpubm <- function(x, i, drop=TRUE)
# {
#     if (!is.numeric(i) & !is.character(i) & !is.logical(i))
#         stop("row indices must be numeric, logical, or character vectors.")
#     if (is.character(i))
#         if (is.null(rownames(x))) stop("row names do not exist.")
#     else i <- mmap(i, rownames(x))
#     if (is.logical(i)) {
#         if (length(i) != nrow(x))
#             stop("row vector length must match the number of rows of the matrix.")
#         i <- which(i)
#     }
#     tempi <- CCleanIndices(as.double(i), as.double(nrow(x)))
#     if (is.null(tempi[[1]])) stop("Illegal row index usage in extraction.\n")
#     if (tempi[[1]]) i <- tempi[[2]]
#     
#     retList <- GetMatrixRows(x@address, as.double(i))
#     mat = .addDimnames(retList, length(i), ncol(x), drop)
#     return(mat)
# }
# 
# GetAll.gpubm <- function(x, drop=TRUE)
# {
#     retList <- GetMatrixAll(x@address)
#     mat = .addDimnames(retList, nrow(x), ncol(x), drop)
#     return(mat)
# }


# #' @export
# setMethod("[",
#           signature(x = "gpuMatrix", drop = "missing"),
#           function(x, i, j) return(GetElements.gpubm(x, i, j)))

# #' @export
# setMethod("[",
#           signature(x = "gpuMatrix", drop = "logical"),
#           function(x, i, j, drop) return(GetElements.gpubm(x, i, j, drop)))
# 
# #' @export
# setMethod("[",
#           signature(x = "gpuMatrix", i="missing", drop = "missing"),
#           function(x, j) return(GetCols.gpubm(x, j)))
# 
# #' @export
# setMethod("[",
#           signature(x = "gpuMatrix", i="missing", drop = "logical"),
#           function(x, j, drop) return(GetCols.gpubm(x, j, drop)))
# 
# #' @export
# setMethod("[",
#           signature(x = "gpuMatrix", j="missing", drop = "missing"),
#           function(x, i) return(GetRows.gpubm(x, i)))
# 
# #' @export
# setMethod("[",
#           signature(x = "gpuMatrix", j="missing", drop = "logical"),
#           function(x, i, drop) return(GetRows.gpubm(x, i, drop)))
# 
# #' @export
# setMethod("[",
#           signature(x = "gpuMatrix", i="missing", j="missing", drop = "missing"),
#           function(x) return(GetAll.gpubm(x)))
# 
# #' @export
# setMethod("[",
#           signature(x = "gpuMatrix", i="missing", j="missing", drop = "logical"),
#           function(x, drop) return(GetAll.gpubm(x, drop)))

# Function contributed by Peter Haverty at Genentech.
# #' @export
# setMethod('[',
#           signature(x = "gpuMatrix",i="matrix",j="missing",drop="missing"),
#           function(x, i) return(GetIndivElements.gpubm(x, i)))

# setMethod('gpuMatrix', 
#           signature(data = 'matrix'),
#           function(data, ncol=NA, nrow=NA, type=NULL){
#               
#               if(is.na(nrow)) nr <- nrow(data) else nr <- nrow
#               if(is.na(ncol)) nc <- ncol(data) else nc <- ncol
#               dn <- dimnames(data)
#               data = switch(typeof(data),
#                             integer = {
#                                 new("igpuMatrix", 
#                                     object=data,
#                                     Dim = as.integer(c(nr,nc)),
#                                     Dimnames = if(is.null(dn)){
#                                         list(NULL,NULL)
#                                     }else{
#                                         dn
#                                     }
#                                 )
#                             },
#                             stop("this is an unrecognized 
#                                  or unimplemented data type")
#               )
#               
#               return(data)
#           },
#           valueClass = "gpuMatrix")


# setMethod('gpuMatrix', 
#           signature(data = 'integer'),
#           function(data, ncol=NA, nrow=NA, type=NULL, byrow=FALSE){
#               
#               if(is_na(ncol)) stop("must define number of columns")
#               if(is.na(nrow)) stop("must define number of rows")
#               data = switch(typeof(data),
#                             integer = {
#                                 new("igpuMatrix", 
#                                     object=matrix(data, 
#                                                   nrow=nrow,
#                                                   ncol=ncol,
#                                                   byrow=T
#                                     ),
#                                     Dim = as.integer(c(nrow,ncol)),
#                                     Dimnames = list(NULL,NULL)
#                                 )
#                             },
#                             stop("this is an unrecognized 
#                                  or unimplemented data type")
#               )
#               
#               return(data)
#           },
#           valueClass = "gpuMatrix")

# #' @export
# gpuMatrix <- function(data = NA, ncol=NA, nrow=NA, type='integer'){
#     if(is(data, "matrix")){
#         if(is.na(nrow)) nr <- nrow(data) else nr <- nrow
#         if(is.na(ncol)) nc <- ncol(data) else nc <- ncol
#         dn <- dimnames(data)
#         data = switch(typeof(data),
#                       integer = {
#                           new("igpuMatrix", 
#                               object=data,
#                               Dim = as.integer(c(nr,nc)),
#                               Dimnames = if(is.null(dn)){
#                                   list(NULL,NULL)
#                               }else{
#                                   dn
#                               }
#                           )
#                       },
#                       stop("this is an unrecognized 
#                                  or unimplemented data type")
#         )
#     }else{
#         print(class(data))
#         stop("Unable to convert object to gpuMatrix")
#     }
#     return(data)
# }