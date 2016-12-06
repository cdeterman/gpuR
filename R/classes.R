
setClassUnion("gpuRmatrix", c("gpuMatrix", "vclMatrix"))

setClass("gpuQR", 
         slots = c(qr = "gpuRmatrix", betas = "numeric")
)