\name{NEWS}
\title{News for Package 'gpuR'}

\section{Unreleased Version 1.1.0}{
    \itemize{
        \item New Features:
        \itemize{
            \item Internal class structure simplified
            \item Scalar Arith methods (e.g. 2*gpuMatrix)
            \item Unary '-' operator (i.e. -gpuMatrix)
            \item Slice methods to point to vector subsets
            \item Block methods to point to matrix subsets
            \item deepcopy method to explicitly copy object contents
            \item 'log','log10', & 'exp' methods for vector objects
            \item 'abs' method added for all objects
            \item 'max' & 'min' method for gpuVector/vclVector objects
            \item 'cbind' & 'rbind' methods for gpuMatrix objects
        }
        \item Features in Progress:
        \itemize{
            \item Additional matrix functions (QR-decompsition, SVD)
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
