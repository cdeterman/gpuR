\name{NEWS}
\title{News for Package 'gpuR'}
\newcommand{\ghpr}{\href{https://github.com/cdeterman/gpuR/pull/#1}{##1}}
\newcommand{\ghit}{\href{https://github.com/cdeterman/gpuR/issues/#1}{##1}}

\section{Version 2.0.2}{
    \itemize{
        \item API Changes:
        \itemize{
            \item The \code{solve} method for \code{gpuMatrix} objects now returns a \code{gpuMatrix} to remain consistent with the overall gpuR API \ghit{108}
            \item The default numeric type has been changed to 'float' to allow all devices to use the functions herein by default as is also generally faster than 'double' which is not always required. \ghit{92}
        }
        \item New Features:
        \itemize{
            \item Added matrix-vector product for \code{gpuMatrix}/\code{gpuVector} objects \ghit{98}
            \item The deprecation warnings often returned from \code{custom_opencl} are now suppressed to avoid end-user confusion \ghit{103}
            \item Implemented \code{tcrossprod} methods for \code{gpuVector}/\code{vclVector} objects \ghit{112}
            \item Improved error message when referring to queues \ghit{119}
            \item \code{custom_opencl} now allows for kernels to call internally defined functions \ghit{129}
            \item Added `inplace` operations for `exp`, `abs` \ghit{141}
        }
        \item Bug Fixes:
        \itemize{
            \item GEMV output dimensions have been fixed \ghit{97}
            \item Fixed bug when using \code{custom_opencl} with multiple scalars \ghit{118}
            \item Fixed bug in \code{custom_opencl} which omitted pragma allowing for double precision in special cases \ghit{123}
            \item Fixed bug in \code{dist} with respect to \code{gpuMatrix} objects exceeding dimensions of 128 \ghit{131}
            \item Fixed bug in integer GEMM kernel observed in larger matrices \ghit{140}
            \item Fixed bug in restricting S4 methods in recent R versions \ghit{152}
        }
        \item Changes in gpuR Documentation:
        \itemize{
            \item Updated \code{custom_opencl} documentation \ghit{120}
            \item Updated the INSTALL file \ghit{101}
        }
    }
}

\section{Version 2.0.0}{
    \itemize{
        \item API Changes:
        \itemize{
            \item deviceType, gpuInfo, cpuInfo not longer accepts 'platform_idx' parameter as OpenCL contexts cannot contain more than one platform.
        }
        \item New Features:
        \itemize{
            \item Added functionality to create custom OpenCL functions from user provided kernels
            \item Added `synchronize` function to assure completion of device calls (necessary for benchmarking)
            \item Added determinant function (det)
            \item Allow for gpuR object - base object interaction (e.g. vclMatrix * matrix)
            \item Added `inplace` function for 'inplace' operations.  These operations include `+`, `-`, `*`, `/`, `sin`, `asin`, `sinh`, `cos`, `acos`, `cosh`, `tan`, `atan`, `tanh`.
            \item Added `sqrt`, `sum`, `sign`,`pmin`, and `pmax` functions
            \item Methods to pass two gpuR matrix objects to `cov`
            \item Added `norm` method
            \item Added gpuRmatrix/gpuRvector Arith `+`,`-` methods
        }
        \item Bug Fixes:
        \itemize{
            \item Fixed incorrect device info when using different contexts
            \item Fixed Integer Matrix Multiplication
            \item All OpenCL devices will be initialized on startup (previous version occasionally would omit some devices)
        }
        \item Features in Progress:
        \itemize{
            \item Parallel GPU functionality
            \item Implementing 'complex' data type
            \item Additional indexing functionality
            \item Additional correlation methods
            \item Additional gpuR - `base R` methods
            \item gpu-vcl object methods
        }
    }
}

\section{Version 1.2.0}{
    \itemize{
        \item New Features:
        \itemize{
            \item Automatically detect available SDK on install if available
            \item Simplified installation to build OpenCL ICD when no SDK installed (thanks Yixuan Qui)
            \item Control over individual OpenCL contexts to allow user to choose device to use
            \item Added as.* methods for vclMatrix/Vector and gpuMatrix/Vector objects
            \item Added str method for matrix objects
            \item Added length method for matrix objects
            \item Added solve method for square vclMatrix objects
            \item Added QR-decompsition, SVD, Cholesky for square gpuMatrix/vclMatrix objects
            \item Added diag and diag<- method for matix objects
            \item Added colnames and colnames<- methods
        }
        \item Features in Progress:
        \itemize{
            \item 'device type' checks (i.e. deal with CPU & GPU objects), perhaps detect type from object?
            \item Parallel GPU functionality
            \item In-place operations
            \item Accept custom OpenCL kernels
        }
    }
}

\section{Version 1.1.2}{
    \itemize{
        \item New Features:
        \itemize{
            \item fixed major bugs with cbind/rbind and deepcopy reflecting new class architectures
        }
        \item Features in Progress:
        \itemize{
            \item Additional matrix functions (QR-decompsition, SVD, Cholesky, solve)
            \item 'device type' checks (i.e. deal with CPU & GPU objects), perhaps detect type from object?
            \item Control over OpenCL contexts for mutliple GPUs
        }
    }
}

\section{Version 1.1.0}{
    \itemize{
        \item New Features:
        \itemize{
            \item Internal class structure simplified
            \item Scalar Arith methods (e.g. 2*X) for gpuMatrix/vclMatrix objects
            \item Unary '-' operator (i.e. -X) for gpuMatrix/vclMatrix objects 
            \item Slice methods to point to gpuVector/vclVector subsets
            \item Block methods to point to gpuMatrix/vclMatrix subsets
            \item deepcopy method to explicitly copy object contents
            \item 'log','log10', & 'exp' methods for vector objects
            \item 'abs' method added for all objects
            \item 'max' & 'min' method for all objects
            \item 'cbind' & 'rbind' methods for gpuMatrix/vclMatrix objects
            \item 'distance' method for pairwise distances
            \item 'squared euclidean' distance added ('sqEuclidean')
            \item 't' method added
        }
        \item Features in Progress:
        \itemize{
            \item Additional matrix functions (QR-decompsition, SVD, Cholesky, solve)
            \item 'device type' checks (i.e. deal with CPU & GPU objects), perhaps detect type from object?
        }
    }
}

\section{Initial Release version 1.0.0}{
    \itemize{
        \item Implemented Features:
        \itemize{
            \item gpu* classes for vector and matrix objects
            \item vcl* classes for matrix objects
            \item Support for 'integer', 'float' and 'double' data types
            \item gpuVector * vclVector functions (addition, subtraction)
            \item gpuMatrix & vclMatrix functions (multiplication, addition, subtraction)
            \item gpuMatrix eigen decomposition (values and vectors)
            \item trig functions for gpu* & vcl* classes
            \item print method for gpuMatrix & vclMatrix objects
            \item dist method gpuMatrix & vclMatrix objects
        }
        \item Features in Progress:
        \itemize{
            \item Additional matrix functions (QR-decompsition, SVD)
        }
    }
}
