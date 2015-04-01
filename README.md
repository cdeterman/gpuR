If installing with Rstudio it won't recongize your CUDA_HOME by default.  
You must update your ${R_HOME}/etc/Renviron to include the variable in order 
for Rstudio to find it.

# Things to Do
1. Create configure file for .Renviron and possibly makevars?
2. Package OpenCL headers in package (similar to ROpenCL?)?