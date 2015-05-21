\name{NEWS}
\title{News for Package 'gpuR'}

\section{Initial Release version 1.0.0 (Unreleased)}{
    \itemize{
        \item Implemented Features:
        \itemize{
            \item gpu* classes for vector, matrix, and
            big.matrix objects
            \item Support for 'integer', 'float' and 'double' data types
            \item gpuVector functions (addition, subtraction)
            \item gpuMatrix functions (multiplication, addition, subtraction)
            \item gpuBigMatrix function (multiplication, addition, subtraction)
        }
        \item Features in Progress:
        \itemize{
            \item Additional matrix functions (QR-decompsition, SVD)
            \item Implement CUDA code if NVIDIA GPU being used
        }
    }
}