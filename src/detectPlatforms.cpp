#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "cl_helpers.hpp"

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
    
