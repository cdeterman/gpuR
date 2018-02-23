

splitAt <- function(x, pos) unname(split(x, cumsum(seq_along(x) %in% pos)))

#' @title Setup OpenCL Arguments
#' @description Generates a \code{data.frame} of argument definitions
#' for use in \code{\link{custom_opencl}}
#' @param objects character vector of gpuR objects to be passed
#' @param intents character vector specifying 'intent' of gpuR objects.
#' options include \code{"IN"},\code{"OUT"},\code{"INOUT"}
#' @param queues list of equal length to \code{"objects"} where each element
#' @param kernel_maps The corresponding arguments names in the provided OpenCL kernel
#' corresponds to the gpuR objects passed and contains a character vector of
#' which kernels the object will be enqueued.
#' @importFrom assertive assert_is_character assert_all_are_same_length assert_is_list
#' @export
setup_opencl <- function(objects, intents, queues, kernel_maps = NULL){

    # make sure character vectors
    assert_is_character(objects)
    assert_is_character(intents)

    # make sure queues is a list
    assert_is_list(queues)

    # must define all object intents
    assert_are_same_length(objects, intents)
    assert_are_same_length(objects, queues)

    # make sure defining possible objects
    assert_all_are_true(objects %in% c('gpuVector', 'vclVector', 'gpuMatrix', 'vclMatrix', 'scalar'))
    assert_all_are_true(intents %in% c("IN", "OUT", "INOUT"))

    if(is.null(kernel_maps) & is.null(names(objects))){
        stop("Either 'objects' must have names corresponding to kernel arguments
             or kernel_maps must be defined")
    }

    out <- vector("list", length = length(objects))

    mappings <- if(is.null(kernel_maps)) names(objects) else kernel_maps

    for(o in seq_along(objects)){
        out[[o]] <- c(objects[o], intents[o], paste0(queues[[o]], collapse = ","), mappings[[o]])
    }

    out <- do.call('rbind', out)
    dimnames(out) <- NULL
    colnames(out) <- c("object", "intents", "queues", "map")
    out <- as.data.frame(out, stringsAsFactors = FALSE)
    return(out)
}

#' @title Custom OpenCL Kernels
#' @description Compile a custom function using a user provided OpenCL kernel
#' @param kernel path to OpenCL kernel file
#' @param cl_args A \code{data.frame} that contains argument definitions.
#' Provided by \code{\link{setup_opencl}}
#' @param type The precision on which the kernel is compiled.  Options include
#' \code{"int"}, \code{"float"}, and \code{"double"}
#' @importFrom assertive assert_is_character
#' @importFrom tools file_path_sans_ext
#' @importFrom Rcpp sourceCpp
#' @export
custom_opencl <- function(kernel, cl_args, type){

    # make sure character vectors
    assert_is_character(type)
    
    if(!type %in% c("integer", "float", "double")){
        stop("type not recognized")
    }
    
    type <- if(type == "integer") "int" else type

    # copy base_base_custom_opencl.cpp to tmp directory
    # using cwd for development paste0(getwd(), "/salamander.cpp") - tempfile()
    
    ocl_shell <- system.file("src", package = "gpuR")
    ocl_file <- paste0(tempfile(), '.cpp')
    # ocl_file <- paste0(getwd(), "/salamander.cpp")
    tryCatch({
        invisible(
            file.copy(paste0(ocl_shell, '/base_custom_opencl.cpp'),
                      ocl_file)
            )
    },
    warning = function(w) {
        warning(w)
        stop("copying the base file failed, see warning message below")
    })


    myfile <- readLines(ocl_file)

    # set input arguments
    input_args <- sapply(1:nrow(cl_args), function(x) {
        id <- cl_args[x,"map"]
        suffix <- if(x == nrow(cl_args)) "_" else "_,"
        if(cl_args[x,"object"] != "scalar"){
            paste0("SEXP ptr", id, suffix)
        }else{
            paste0("SEXP ", id, suffix)
        }

    })

    # list of input objects for later use
    input_objs <- sapply(input_args, function(x){
        # y <- unlist(strsplit(input_args[3], " "))
        unlist(lapply(strsplit(x, " "), function(y) substr(y[[length(y)]], 0, nchar(y[[length(y)]])-1)))
    }, USE.NAMES = FALSE)

    # context index
    # grab first cl argument
    cl_arg <- cl_args[cl_args$object %in% c("gpuMatrix", "gpuVector", "vclMatrix", "vclVector"),]
    context_index_line <- paste('const int ctx_id = as<int>(s4', cl_arg[1,"map"], '.slot(".context_index")) - 1;', sep = "")

    s4_lines <- sapply(1:nrow(cl_arg), function(x) {
        id <- cl_args[x,"map"]
        paste0("Rcpp::S4 s4",
               id,
               "(ptr",
               id,
               "_);")
    })

    id <- cl_args[1,"map"]
    context_lines <- c(paste0('viennacl::ocl::context ctx = vcl_',
                           id, '->handle().opencl_handle().context();'),
                    paste0('cl_context my_context = vcl_',
                           id, '->handle().opencl_handle().context().handle().get();'),
                    paste0('cl_device_id my_device = vcl_',
                           id, '->handle().opencl_handle().context().devices()[0].id();'),
                    paste0('cl_command_queue queue = vcl_',
                           id, '->handle().opencl_handle().context().get_queue().handle().get();')
    )
    
    
    # import lines
    import_lines <- sapply(1:nrow(cl_args), function(x) {
        switch(cl_args[x,"object"],
               "gpuVector" = {
                   id <- cl_args[x,"map"]
                   paste0("std::shared_ptr<viennacl::vector_base<",
                          type,
                          "> > vcl_",
                          id,
                          " = getVCLVecptr<",
                          type,
                          ">(s4",
                          id,
                          '.slot("address"), false, ctx_id);')
               },
               "gpuMatrix" = {
                   id <- cl_args[x,"map"]
                   paste0("std::shared_ptr<viennacl::matrix<",
                          type,
                          "> > vcl_",
                          id,
                          " = getVCLptr<",
                          type,
                          ">(s4",
                          id,
                          '.slot("address"), false, ctx_id);')
               },
               "vclVector" = {
                   id <- cl_args[x,"map"]
                   paste0("std::shared_ptr<viennacl::vector_base<",
                          type,
                          "> > vcl_",
                          id,
                          " = getVCLVecptr<",
                          type,
                          ">(s4",
                          id,
                          '.slot("address"), true, ctx_id);')
               },
               "vclMatrix" = {
                   id <- cl_args[x,"map"]
                   paste0("std::shared_ptr<viennacl::matrix<",
                          type,
                          "> > vcl_",
                          id,
                          " = getVCLptr<",
                          type,
                          ">(s4",
                          id,
                          '.slot("address"), true, ctx_id);')
               },
               "scalar" = {
                   id <- cl_args[x,"map"]
                   paste0(type,
                          " ",
                          id,
                          "= as<",
                          type,
                          ">(",
                          id,
                          "_ ",
                          ");")
               }
        )
    })

    # import dims
    import_dims <- lapply(1:nrow(cl_args), function(x){
      switch(cl_args[x,"object"],
             "gpuVector" = {
                 id <- cl_args[x,"map"]
                 paste0("unsigned int ",
                        id,
                        "_size = vcl_",
                        id,
                        "->size();")
             },
             "gpuMatrix" = {
                 id <- cl_args[x,"map"]
                 r <- paste0("unsigned int ",
                             id,
                             "_size1 = vcl_",
                             id,
                             "->size1();")
                 ri <- paste0("unsigned int ",
                             id,
                             "_internal_size1 = vcl_",
                             id,
                             "->internal_size1();")
                 c <- paste0("unsigned int ",
                             id,
                             "_size2 = vcl_",
                             id,
                             "->size2();")
                 ci <- paste0("unsigned int ",
                             id,
                             "_internal_size2 = vcl_",
                             id,
                             "->internal_size2();")
                 return(c(r, ri, c, ci))
             },
             "vclVector" = {
                 id <- cl_args[x,"map"]
                 paste0("unsigned int ",
                        id,
                        "_size = vcl_",
                        id,
                        "->size();")
             },
             "vclMatrix" = {
                 id <- cl_args[x,"map"]
                 r <- paste0("unsigned int ",
                             id,
                             "_size1 = vcl_",
                             id,
                             "->size1();")
                 ri <- paste0("unsigned int ",
                              id,
                              "_internal_size1 = vcl_",
                              id,
                              "->internal_size1();")
                 c <- paste0("unsigned int ",
                             id,
                             "_size2 = vcl_",
                             id,
                             "->size2();")
                 ci <- paste0("unsigned int ",
                              id,
                              "_internal_size2 = vcl_",
                              id,
                              "->internal_size2();")
                 return(c(r, ri, c, ci))
             })
    })

    dim_objs <- sapply(import_dims, function(x){
        if(!is.null(x)){
            unlist(lapply(strsplit(x, " "), function(y) y[[3]]))    
        }
    })

    # read in kernel
    src <- readLines(kernel, file.info(kernel)$size)
    src_quoted <- sprintf('"%s\\\\n"', src)

    # remove blank lines
    src <- src[sapply(src, function(x) length(grep("^\\s*$", x)) == 0)]

    # separate kernels
    kernels <- splitAt(src, which(sapply(src, function(x) grepl("\\_\\_kernel", x, perl = TRUE), USE.NAMES = FALSE)))

    # get kernel names
    knames <- sapply(kernels, function(x) {
        gsub("\\(.*", "", unlist(strsplit(x[1], " "))[3])
    }, USE.NAMES = FALSE)

    knames_line <- paste0('Rcpp::StringVector kernel_name("',paste(knames, collapse = '","'), '");')

    kernel_args <- sapply(kernels, function(x){
        unlist(strsplit(paste(x, collapse = ' '), '[()]'))[2]
    })


    # import kernels
    #viennacl::ocl::kernel & update_kk = my_prog.get_kernel("update_kk");
    import_kernels <- sapply(knames, function(x) {
        paste0("viennacl::ocl::kernel & ", x, ' = my_prog.get_kernel("', x, '");\n')
    }, USE.NAMES = FALSE)

    # get counts of global ids to set
    globals <- sapply(kernels, function(x) {
        sum(sapply(regmatches(x, gregexpr("get_global_id", x)), length))
    }, USE.NAMES = FALSE)

    # # get preferred multiple of work sizes
    # # readChar(kernel, file.info(kernel)$size)
    # pwg <- sapply(seq_along(kernels), function(x){
    #     gpuR:::preferred_wg_size(paste(kernels[[x]], collapse = "\r\n"), knames[x], 0L)
    # })


    # global lines
    import_global <- sapply(seq_along(globals), function(x) {
        kname <- knames[x]
        kdim <- as.character(globals[x])

        switch(kdim,
               "0" = {paste0(kname, ".global_work_size(0, 1);")},
               "1" = {
                   objects_in_kernel <- cl_args[which(sapply(cl_args[, "queues"], function(q) grepl(kname, q))),]
                   if(any(grepl("Vector", objects_in_kernel[, "object"]))){
                       vecs <- paste0(objects_in_kernel[grepl("Vector", objects_in_kernel[,"object"]), "map"], "_size")
                   }else{
                       vecs <- NULL
                   }

                   if(any(grepl("Matrix", objects_in_kernel[, "object"]))){
                       mats <- paste0(objects_in_kernel[grepl("Matrix", objects_in_kernel[,"object"]), "map"], "_internal_size1")
                   }else{
                       mats <- NULL
                   }

                   paste0(kname, ".global_work_size(0, roundUp(std::max({", paste(c(vecs, mats), collapse = ","), "}), max_local_size[0]));")
               },
               "2" = {
                   objects_in_kernel <- cl_args[which(sapply(cl_args[, "queues"], function(q) grepl(kname, q))),]
                   if(any(grepl("Vector", objects_in_kernel[, "object"]))){
                       vecs <- paste0(objects_in_kernel[grepl("Vector", objects_in_kernel[,"object"]), "map"], "_size")
                   }else{
                       vecs <- NULL
                   }

                   if(any(grepl("Matrix", objects_in_kernel[, "object"]))){
                       mats1 <- paste0(objects_in_kernel[grepl("Matrix", objects_in_kernel[,"object"]), "map"], "_internal_size1")
                       mats2 <- paste0(objects_in_kernel[grepl("Matrix", objects_in_kernel[,"object"]), "map"], "_internal_size2")
                   }else{
                       mats1 <- NULL
                       mats2 <- NULL
                   }

                   c(paste0(kname, ".global_work_size(0, roundUp(std::max({", paste(c(vecs, mats1), collapse = ","), "}), sqrt(max_local_size[0])));"),
                     paste0(kname, ".global_work_size(1, roundUp(std::max({", paste(c(vecs, mats2), collapse = ","), "}), sqrt(max_local_size[0])));"))
               },
               "3" = stop("3 dimensional not yet implemented"),
               stop("unrecognized dimension")
        )
    }, USE.NAMES = FALSE)

    # local lines
    import_local <- sapply(seq_along(globals), function(x) {
        kname <- knames[x]
        kdim <- as.character(globals[x])

        switch(kdim,
               "0" = {paste0(kname, ".local_work_size(0, 1);")},
               "1" = {
                   objects_in_kernel <- cl_args[which(sapply(cl_args[, "queues"], function(q) grepl(kname, q))),]
                   if(any(grepl("Vector", objects_in_kernel[, "object"]))){
                       vecs <- paste0(objects_in_kernel[grepl("Vector", objects_in_kernel[,"object"]), "map"], "_size")
                   }else{
                       vecs <- NULL
                   }

                   if(any(grepl("Matrix", objects_in_kernel[, "object"]))){
                       mats <- paste0(objects_in_kernel[grepl("Matrix", objects_in_kernel[,"object"]), "map"], "internal_size1")
                   }else{
                       mats <- NULL
                   }

                   paste0(kname, ".local_work_size(0, max_local_size[0]);")
               },
               "2" = {
                   objects_in_kernel <- cl_args[which(sapply(cl_args[, "queues"], function(q) grepl(kname, q))),]
                   if(any(grepl("Vector", objects_in_kernel[, "object"]))){
                       vecs <- paste0(objects_in_kernel[grepl("Vector", objects_in_kernel[,"object"]), "map"], "_size")
                   }else{
                       vecs <- NULL
                   }

                   if(any(grepl("Matrix", objects_in_kernel[, "object"]))){
                       mats1 <- paste0(objects_in_kernel[grepl("Matrix", objects_in_kernel[,"object"]), "map"], "internal_size1")
                       mats2 <- paste0(objects_in_kernel[grepl("Matrix", objects_in_kernel[,"object"]), "map"], "internal_size2")
                   }else{
                       mats1 <- NULL
                       mats2 <- NULL
                   }

                   c(paste0(kname, ".local_work_size(0, sqrt(max_local_size[0]));"),
                     paste0(kname, ".local_work_size(1, sqrt(max_local_size[0]));"))
               },
               "3" = stop("3 dimensional not yet implemented"),
               stop("unrecognized dimension")
        )
    }, USE.NAMES = FALSE)


    # queue lines
    # viennacl::ocl::enqueue(my_kernel(*vcl_A, *vcl_B, value, M, P, M_internal));
    k_args <- lapply(kernel_args, function(x) unlist(strsplit(x, ",")))
    k_args <- lapply(k_args, function(x) {
        unlist(lapply(strsplit(x, "\\bdouble|\\bfloat|\\bint"), function(y){
            gsub(" ", "", y[[length(y)]])
        }))
    })

    # get pointer arguments (i.e. opencl buffers)
    ptr_args <- lapply(k_args, function(x) x[grepl('\\*', x)])
    
    # make sure input arguments match the kernel buffers
    if(!all(cl_arg[,"map"] %in% gsub("\\*", "", unlist(ptr_args)))){
        stop("Not all OpenCL buffers from kernel are mapped")
    }
    
    # non-gpu objects (not opencl buffers)
    non_cl_objs <- lapply(k_args, function(x) x[!grepl('\\*', x)])

    # probably want additional documentation on what this error actually means
    arg_checks <- unlist(lapply(non_cl_objs, function(x) any(!x %in% c(dim_objs, input_objs))))
    if(any(arg_checks)){
        stop("Non OpenCL buffer kernel arguments don't match to initialized objects.")
    }

    # create enqueue lines(s)
    enqueue_lines <- sapply(seq_along(knames), function(k){
        objects_in_kernel <- cl_args[which(sapply(cl_args[, "queues"], function(q) grepl(knames[k], q))),]
        cpp_objs <- paste0("*vcl_", objects_in_kernel[objects_in_kernel[,"object"] != "scalar","map"])
        cl_objs <- paste0("*", objects_in_kernel[objects_in_kernel[,"object"] != "scalar","map"])
        cl_objs <- c(cl_objs, objects_in_kernel[objects_in_kernel[,"object"] == "scalar","map"])
        
        internal_cpp_objs <- non_cl_objs[[k]]
        
        # my_objs <- c(cpp_objs, internal_cpp_objs)
        my_vcl_objs <- c(cpp_objs, internal_cpp_objs)
        my_objs <- c(cl_objs, internal_cpp_objs)
        
        paste0("viennacl::ocl::enqueue(", knames[k], "(",
               paste(
                   paste(my_vcl_objs[match(k_args[[k]], my_objs)], collapse = ","),
                   sep = ","),
               "));")
    })

    if(any(cl_args[,"object"] %in% c("gpuVector", "gpuMatrix"))){
        gpu_objs <- cl_args[cl_args$object %in% c("gpuVector", "gpuMatrix"),]
        if(any(cl_args[,"intents"] %in% c("INOUT", "OUT"))){

            tmp <- gpu_objs[gpu_objs[,"intents"] %in% c("INOUT", "OUT"),]

            frame <- "if(!OBJECTisVCL){
                Rcpp::XPtr<dynEigenMat<T> > ptrOBJECT(ptrOBJECT_);

                // copy device data back to CPU
                ptrOBJECT->to_host(*vcl_OBJECT);
                ptrOBJECT->release_device();
            }"
            frame <- gsub("<T>", paste0("<", type, ">"), frame)

            out_lines <- vector("list", length = nrow(tmp))
            for(i in 1:nrow(tmp)){
                out_lines[[i]] <- gsub("OBJECT", tmp[i,"map"], frame)
            }
            out_lines[[2]] <- out_lines[[1]]
            out_lines <- do.call("paste", list(out_lines, collapse = "\n\n"))

        }
    }else{
        out_lines <- NULL
    }

    myfile[grepl("CPP_NAME", myfile)] <- gsub("CPP_NAME", basename(file_path_sans_ext(kernel)), myfile[grepl("CPP_NAME", myfile)])
    myfile[grepl("MY_ARGS", myfile)] <- gsub("MY_ARGS", paste(input_args, collapse="\n"), myfile[grepl("MY_ARGS", myfile)])
    myfile[grepl("MY_KERNEL_NAMES", myfile)] <- gsub("MY_KERNEL_NAMES", knames_line, myfile[grepl("MY_KERNEL_NAMES", myfile)])
    myfile[grepl("MY_S4", myfile)] <- gsub("MY_S4", paste(s4_lines, collapse="\n"), myfile[grepl("MY_S4", myfile)])
    myfile[grepl("MY_CTX_ID", myfile)] <- gsub("MY_CTX_ID", context_index_line, myfile[grepl("MY_CTX_ID", myfile)])
    myfile[grepl("MY_CONTEXT", myfile)] <- gsub("MY_CONTEXT", paste(context_lines, collapse="\n"), myfile[grepl("MY_CONTEXT", myfile)])
    myfile[grepl("MY_KERNEL_SRC", myfile)] <- gsub("MY_KERNEL_SRC", paste(src_quoted, collapse="\n"), myfile[grepl("MY_KERNEL_SRC", myfile)])
    myfile[grepl("MY_DEFINES", myfile)] <- gsub("MY_DEFINES", paste(import_lines, collapse = "\n"), myfile[grepl("MY_DEFINES", myfile)])
    myfile[grepl("MY_DIMS", myfile)] <- gsub("MY_DIMS", paste(unlist(import_dims), collapse = "\n"), myfile[grepl("MY_DIMS", myfile)])
    myfile[grepl("MY_KERNELS", myfile)] <- gsub("MY_KERNELS", paste(import_kernels, collapse = "\n"), myfile[grepl("MY_KERNELS", myfile)])
    myfile[grepl("MY_GLOBALS", myfile)] <- gsub("MY_GLOBALS", paste(import_global, collapse = "\n"), myfile[grepl("MY_GLOBALS", myfile)])
    myfile[grepl("MY_LOCALS", myfile)] <- gsub("MY_LOCALS", paste(import_local, collapse = "\n"), myfile[grepl("MY_LOCALS", myfile)])
    myfile[grepl("MY_QUEUES", myfile)] <- gsub("MY_QUEUES", paste(enqueue_lines, collapse = "\n"), myfile[grepl("MY_QUEUES", myfile)])
    myfile[grepl("MY_OUT", myfile)] <- gsub("MY_OUT", if(is.null(out_lines)) "" else out_lines, myfile[grepl("MY_OUT", myfile)])

    writeLines(myfile, ocl_file)
    
    # setup environment for compiling
    os <- Sys.info()['sysname']
    pkg_inc <- system.file("include", package = "gpuR")
    
    switch(os,
           "Windows" = {
               arch <- if(R.Version()[["arch"]] == "x86_64") "x64" else "i386"
               LIBS <- "-LPATH/loader/ARCH -lOpenCL -Wl,-rpath,PATH/loader/ARCH"
               LIBS <- gsub("PATH", paste('"', pkg_inc, '"', sep = ""), LIBS)
               LIBS <- gsub("ARCH", arch, LIBS)
               Sys.setenv(PKG_LIBS=LIBS)
           },
           "Darwin" = {
               Sys.setenv(PKG_LIBS="-framework OpenCL")
           },
           {
               Sys.setenv(PKG_LIBS="-lOpenCL")
           })
    
    sourceCpp(ocl_file)
}


