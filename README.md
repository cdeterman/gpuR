# gpuR
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

# Existing BUGS!!!
1. Curiously, multiplying two `gpuMatrix` objects works correctly but when
I try to run all my tests the program will segfault.  Some initial debugging
shows that the program is crashing when the OpenCL program is being built
(`program.build(devices)`) **Could definitely use help by anyone very familiar 
with OpenCL code structure**.  Please see Issue#1 for further details.

# INSTALL (also see the INSTALL file)

The only verified installations at present consisted of using a NVIDIA GTX or
AMD Radeon Graphics Card on a Ubuntu 14.04 system.  The installation 
consisted of:

#### Note, you currently can only have one type installed (NVIDIA or AMD)

### NVIDIA Driver and CUDA/OpenCL
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

### AMD Driver and OpenCL
1. Purge existing fglrx drivers (`sudo sh /usr/share/ati/fglrx_uninstall.sh`)
2. Install current fglrx drivers (`sudo apt-get install fglrx-updates`)
3. Install opencl-headers (`sudo apt-get install opencl-headers`) -- needed
to install clBLAS

### Install clBLAS
1. See INSTALL file

### C++ OpenCL API
**If using an AMD card, ignore this secion.**
You then may need to have the C++ API header file.  if your GPU only
supports OpenCL 1.1.  You can get the hpp file from the 
[Khronos registry](https://www.khronos.org/registry/cl/api/1.1/cl.hpp).  Once
you have downloaded the file you need to put it with the other OpenCL headers.
In the case of NVIDIA, the location of the OpenCL headers is 
`/usr/local/cuda-x.x/include/CL`.  You will then need to modify the Makevars
file (see the current Makevars file for an example).

### Rstudio settings
**If using an AMD card, ignore this secion.**
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
4. Conditional `double` data types if device supports them
5. Implement clBLAS?  Appears to only be for AMD though :(  Other alternatives
appeart to be MAGMA (also AMD).
6. Alternative approach, optimized code for each Vendor Type (NVIDIA, AMD, etc.)