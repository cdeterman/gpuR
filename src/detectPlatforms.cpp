#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#ifdef CL_VERSION_1_3
#undef CL_VERSION_1_3
#endif

#ifdef CL_VERSION_1_2
#undef CL_VERSION_1_2
#endif

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "gpuR/cl_helpers.hpp"

#include <Rcpp.h>

using namespace cl;
using namespace Rcpp;

//' @title Detect Number of Platforms
//' @description Find out how many OpenCL enabled platforms are available.
//' @return An integer value representing the number of platforms available.
//' @seealso \link{detectGPUs}
//' @export
// [[Rcpp::export]]
SEXP detectPlatforms()
{
    // Get available platforms
    std::vector<Platform> platforms;

    getPlatforms(platforms); // cl_helpers.hpp
    
    return(wrap(platforms.size()));
}
    
