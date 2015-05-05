
has_gpu_skip <- function(idx = 1L) {
    assert_is_an_integer(idx)
    gpuCheck <- try(detectGPUs(), silent=TRUE)
    if(class(gpuCheck)[1] == "try-error"){
        skip("No GPUs available")
    }else{
        if (gpuCheck == 0) {
            skip("No GPUs available")
        }
    }
}


