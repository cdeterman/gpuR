
get_os <- function()
{
    os = tolower(Sys.info()["sysname"])
    names(os) = NULL
    
    os
}

CxxFlags = function()
{
    os = get_os()
    
    libs_dir_rel = system.file("include", package="gpuR")
    libs_dir = tools::file_path_as_absolute(libs_dir_rel)
    
    if (os == "linux" || os == "freebsd")
        flags = libs_dir
    else{
        dll.path <- normalizePath(libs_dir)
        dll.path <- utils::shortPathName(dll.path)
        flags <- gsub("\\\\", "/", dll.path)
    }
    
    cat(flags)
}
