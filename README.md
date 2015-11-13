# gpuR

### Build Status
|                 | Build           |
|-----------------|-----------------|
| Linux x86_64    | [![Build Status](https://travis-ci.org/cdeterman/gpuR.png?branch=master)](https://travis-ci.org/cdeterman/gpuR)      |
| OSX             | [![Build Status](https://travis-ci.org/cdeterman/gpuR.png?branch=macosx)](https://travis-ci.org/cdeterman/gpuR)          |
| Windows x86     | [![Appveyor Build Status](https://ci.appveyor.com/api/projects/status/github/cdeterman/gpuR?branch=master)](https://ci.appveyor.com/project/cdeterman/gpuR)     |

Test coverage: [![Coverage Status](https://coveralls.io/repos/cdeterman/gpuR/badge.svg)](https://coveralls.io/r/cdeterman/gpuR?branch=master)

Welcome to my R package for simple GPU computing.  Although there are a few
existing packages to leverage the power of GPU's they are either specific
to one brand (e.g. NVIDIA) or are not very user friendly.  The goal of this
package is to provide the user a very simple R API that can be used with
any GPU (via an OpenCL backend).  This is accomplished by interfacing with the 
ViennaCL library that I have packaged in the R package 
[RViennaCL](http://github.com/cdeterman/RViennaCL).  To make the R API simple,
I created new classes similar to the structure of the 
[Matrix](http://cran.r-project.org/web/packages/Matrix/index.html)
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

I also recommend you read the vignette I included with this package to
get a better understanding of its' capabilities.

```r
vignette("gpuR")
```

Please note, all the functions herein use an OpenCL backend.  If you prefer
to have a CUDA based backend, please see my other package 
[gpuRcuda](http://github.com/cdeterman/gpuRcuda) which is simply an extension
on top of this package where all functions herein are still applicable
but also have the CUDA option available.  

# INSTALL
Please see my [github wiki](https://github.com/cdeterman/gpuR/wiki) for
installation instructions relevant to your operating system.

# Things to Do
1. Obviously more vector functions and matrix implementations
2. My resources limit how much I can test (e.g. OS, GPU vendors).  Would
appreciate any feedback on how the installation and use fairs with other
platforms and GPUs.
2. Would love any suggestions :) (submit in the issues)
