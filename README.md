# bigGPU
Welcome to my R package for simple GPU computing.  Although there are a few
existing packages to leverage the power of GPU's they are either specific
to one brand (e.g. NVIDIA) or are not very user friendly.  The goal of this
package is to provide the user a very simple R API.  This is accomplished by
creating new classes similar to the structure of the [Matrix](http://cran.r-project.org/web/packages/Matrix/index.html)
package.  By doing so, typical methods may be overloaded to make for a very
pleasing sytax.  For example, to perform vector addition the syntax is: 

```r
A <- seq.int(from=0, to=999)
B <- seq.int(from=1000, to=1)
gpuA <- as.gpuVector(A)
gpuB <- as.gpuVector(B)

C <- A + B
Cgpu <- gpuA + gpuB

all(C == gpuC@object)
[1] TRUE
```

# INSTALL
The only verified installation at present is using a NVIDIA Graphics Card
on a Ubuntu 14.04 system.  The installation consisted of:

1. Purge existing nvidia and cuda implementations 
(`sudo apt-get purge cuda* nvidia-*`)
2. Download appropriate CUDA toolkit for the specific card.  You can figure 
this out by first checking which NVIDIA driver is compatible with your card
by searching for it in [NVIDIA's Driver Downloads](http://www.nvidia.com/Download/index.aspx?lang=en-us).
Then check which cuda toolkit is compatible with the driver from this
[Backward Compatibility Table](http://docs.roguewave.com/totalview/8.14.1/html/index.html#page/User_Guides/totalviewug-about-cuda.31.4.html)
The cuda-6.5 toolkit was appropriate for me which you can download from the 
[CUDA toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive).
Once downloaded, run the .run file.
3. Reboot computer
4. Switch to ttyl (Ctrl-Alt-F1)
5. Stop the X server (sudo stop lightdm)
6. Run the cuda run file (`sh cuda_6.5.14_linux_64.run`)
7. Select 'yes' and accept all defaults
8. Required reboot
9. Switch to ttyl, stop X server and run the cuda run file again and select 
'yes' and default for everything (including the driver again)
10. Update `PATH` to include `/usr/local/cuda-6.5/bin` and `LD_LIBRARY_PATH`
to include `/usr/local/cuda-6.5/lib64`
11. Reboot again

If installing with Rstudio it won't recongize your `CUDA_HOME` by default.  
You must currently update your `${R_HOME}/etc/Renviron` to include the variable 
in order for Rstudio to find it.  I am hoping to have the package set this
by default for whichever GPU you are utilizing in the future.

Once all these things are set you should be able to install the package
and begin using your GPU :)

# Things to Do
1. Create configure file for .Renviron and possibly makevars?
2. Package OpenCL headers in package (similar to ROpenCL?)?
3. Obviously more vector functions and matrix implementations