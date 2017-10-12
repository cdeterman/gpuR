
setClassUnion("gpuRmatrix", c("gpuMatrix", "vclMatrix"))
setClassUnion("numericOrInt", c("numeric", "integer"))

setClass("gpuQR", 
         slots = c(qr = "gpuRmatrix", betas = "numeric")
)