\name{NEWS}
\title{News for Package 'gpuR'}


\section{Version 1.1.4}{
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
        }
        \item Features in Progress:
        \itemize{
            \item 'device type' checks (i.e. deal with CPU & GPU objects), perhaps detect type from object?
            \item Parallel GPU functionality
            \item Add diag and diag<- method for matix objects
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
