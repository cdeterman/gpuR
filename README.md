# gpuR
[![Travis-CI Build Status](https://travis-ci.org/cdeterman/gpuR.png?branch=master)](https://travis-ci.org/cdeterman/gpuR)

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
gpuA <- gpuVector(A)
gpuB <- gpuVector(B)

C <- A + B
gpuC <- gpuA + gpuB

all(C == gpuC)
[1] TRUE
```

Please note, all the functions herein use an OpenCL backend.  If you prefer
to have a CUDA based backend, please see my other package 
[gpuRcuda](http://github.com/cdeterman/gpuRcuda) which is simply an extension
on top of this package where all functions herein are still applicable
but also have the CUDA option available.  

# INSTALL (also see the INSTALL file)

The only verified installations at present consisted of using a NVIDIA GTX or
AMD Radeon Graphics Card on a Ubuntu 14.04 system.  The installation 
consisted of:

### Dependencies
1. opencl-headers (shared library)
2. clBLAS (maintained by [ArrayFire](https://github.com/arrayfire/clBLAS))
3. An OpenCL SDK specific to your GPU vender (AMD, NVIDIA, Intel, etc.)

#### Note, you currently can only have one type installed (NVIDIA or AMD)

### NVIDIA Driver and CUDA/OpenCL
#### Up-to-date Card
If you are fortunate enough to have a very recent card that you can
use the most recent drivers.  THis install is much more simple
```
# Install Boost & OpenCL headers
sudo apt-get install opencl-headers

# Install NVIDIA Drivers and CUDA
sudo add-apt-repository -y ppa:xorg-edgers/ppa
sudo apt-get update
sudo apt-get install nvidia-346 nvidia-settings
sudo apt-get install cuda
```

#### Older Card
If you have an older card that doesn't support the newest drivers:

1. Purge any existing nvidia and cuda implementations 
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
```
# Install Boost & OpenCL headers
sudo apt-get install libboost-all-dev opencl-headers

# Install clBLAS
git clone https://github.com/arrayfire/clBLAS.git
cd clBLAS
mkdir build && cd build
cmake ../src -DCMAKE_BUILD_TYPE=Release
make
sudo make install
```

Once all these things are set you should be able to install the package 
and begin using your GPU :)

# Things to Do
1. Obviously more vector functions and matrix implementations
2. Implement clMAGMA?  Probably going to use ViennaCL and avoid clBLAS too.
